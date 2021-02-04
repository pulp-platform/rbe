/*
 * rbe_binconv_sop.sv
 * Gianna Paulin <pauling@iis.ee.ethz.ch>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2019-2021 ETH Zurich, University of Bologna
 * Copyright and related rights are licensed under the Solderpad Hardware
 * License, Version 0.51 (the "License"); you may not use this file except in
 * compliance with the License.  You may obtain a copy of the License at
 * http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
 * or agreed to in writing, software, hardware and materials distributed under
 * this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Depending on ctrl_i.operation_sel perform 1:xnor or 0:and operation on
 * the incoming activation_i and weight_i. popcount_o returns the sum of all
 * `TP` results.
 * With `PIPE_STAGE_LOCATION_*` up to three pipeline stages can be added.
 *
 */

module rbe_binconv_sop #(
  parameter int unsigned TP                    = rbe_package::BINCONV_TP, // number of input elements processed per cycle
  parameter int          PIPE_STAGE_LOCATION_0 = -1, // put 1st pipeline stage, if <0 no pipe stage
  parameter int          PIPE_STAGE_LOCATION_1 = -1, // put 2nd pipeline stage, if <0 no pipe stage
  parameter int          PIPE_STAGE_LOCATION_2 = -1  // put 3rd pipeline stage, if <0 no pipe stage
) (
  // global signals
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   test_mode_i,
  // local enable & clear
  input  logic                   enable_i,
  input  logic                   clear_i,
  // input feat stream + handshake
  hwpe_stream_intf_stream.sink   activation_i,
  // input weight stream + handshake
  hwpe_stream_intf_stream.sink   weight_i,
  // output features + handshake
  hwpe_stream_intf_stream.source popcount_o,
  // control and flags
  input  rbe_package::ctrl_sop_t ctrl_i
);

  /////////////
  // SIGNALS //
  /////////////

  // Streamers
  logic [TP-1:0] feat_data;
  logic [TP-1:0] weight_data;
  logic [TP-1:0] conv_data;

  // Product Data
  logic [TP-1:0] xnor_data;
  logic [TP-1:0] and_data;
  logic [TP-1:0] prod_data;

  // XNOR Reduction Tree
  logic unsigned [$clog2(TP):0][TP-1:0][$clog2(TP):0] reduction_tree;
  logic unsigned [$clog2(TP):0]                       reduction_bitcount;

  logic output_valid;

  logic activation_valid_hs;
  logic weight_valid_hs;


  ////////////////////////////////
  // INPUT STREAMER HANDSHAKING //
  ////////////////////////////////

  always_comb
  begin : ready_propagation
    case({activation_i.valid, weight_i.valid})
      2'b00 : begin
        activation_i.ready = 1'b1;
        weight_i.ready     = 1'b1;
      end
      2'b01 : begin
        activation_i.ready = 1'b1;
        weight_i.ready     = 1'b0;
      end
      2'b10 : begin
        activation_i.ready = 1'b0;
        weight_i.ready     = 1'b1;
      end
      2'b11 : begin
        activation_i.ready = 1'b1;
        weight_i.ready     = 1'b1;
      end
    endcase
  end

  ////////////////////////////////
  // XNOR / AND BINARY PRODUCTS //
  ////////////////////////////////

  // input data assignment
  assign feat_data   = activation_i.data;
  assign weight_data = weight_i.data;

  // compute XNOR / AND binary product and select which is used depending on ctrl
  assign xnor_data = feat_data ~^ weight_data;
  assign and_data  = feat_data  & weight_data;
  assign prod_data = ctrl_i.operation_sel ? xnor_data : and_data;


  ///////////////////////////////////
  // SUM REDUCTION TREE (POPCOUNT) //
  ///////////////////////////////////
  // Reduction tree: counts only the number of 1's (Hamming weight)

  always_comb
  begin
    reduction_tree[0] = '0;
    // first layer
    for(int i=0; i<TP; i++) begin
      if(ctrl_i.inactive_mask[i]) begin
        reduction_tree[0][i] = '0;
      end
      else begin
        reduction_tree[0][i] =  prod_data[i];
      end
    end
  end

  for(genvar jj=1; jj<$clog2(TP)+1; jj++) begin : reduction_tree_layers
    logic unsigned [TP-1:0][$clog2(TP):0] reduction_tree_comb;

    for(genvar ii=0; ii<TP/(2**jj); ii++) begin : reduction_tree_nodes
      assign reduction_tree_comb[ii] = reduction_tree[jj-1][2*ii] + reduction_tree[jj-1][2*ii+1];
    end

    for(genvar ii=TP/(2**jj); ii<TP; ii++) begin : reduction_tree_nils
      assign reduction_tree_comb[ii] = '0;
    end

    // add pipeline registers where needed
    if((jj == PIPE_STAGE_LOCATION_0) || (jj == PIPE_STAGE_LOCATION_1) ||
       (jj == PIPE_STAGE_LOCATION_2)) begin : pipe_stage_gen

      always_ff @(posedge clk_i or negedge rst_ni)
      begin
        if(~rst_ni)
          reduction_tree[jj] <= '0;
        else if(clear_i)
          reduction_tree[jj] <= '0;
        else if(enable_i)
          reduction_tree[jj] <= reduction_tree_comb;
      end

    end
    else begin : nopipe_stage_gen

      assign reduction_tree[jj] = reduction_tree_comb;

    end
  end // reduction_tree_layers

  assign reduction_bitcount = reduction_tree[$clog2(TP)];

  ///////////////////////////////////////
  // PIPELINE CORRECT HANDSHAKE SIGNAL //
  ///////////////////////////////////////

  assign activation_valid_hs = activation_i.valid & activation_i.ready;
  assign weight_valid_hs     = weight_i.valid     & weight_i.ready;

  logic [2:0] handshake_pipe;
  // write enable driven by the pipe stage correct handshake signal
  if (PIPE_STAGE_LOCATION_0 > 0) begin : pipe_stage_gen
    if (PIPE_STAGE_LOCATION_1 > 0) begin : pipe_two_stage_gen
      if (PIPE_STAGE_LOCATION_2 > 0) begin : pipe_three_stage_gen
        // Three pipeline stages
        always_ff @(posedge clk_i or negedge rst_ni)
        begin
          if(~rst_ni)
            handshake_pipe <= '0;
          else if(clear_i)
            handshake_pipe <= '0;
          else if(enable_i) begin
            handshake_pipe[2] <= handshake_pipe[1];
            handshake_pipe[1] <= handshake_pipe[0];
            handshake_pipe[0] <= activation_valid_hs & weight_valid_hs;
          end
        end
        assign output_valid = handshake_pipe[2];
      end
      else begin : pipe_two_stage_gen
        // Two pipeline stages
        always_ff @(posedge clk_i or negedge rst_ni)
        begin
          if(~rst_ni)
            handshake_pipe <= '0;
          else if(clear_i)
            handshake_pipe <= '0;
          else if(enable_i) begin
            handshake_pipe[1] <= handshake_pipe[0];
            handshake_pipe[0] <= activation_valid_hs & weight_valid_hs;
          end
        end
        assign output_valid = handshake_pipe[1];
      end
    end
    else begin : pipe_one_stage_gen
      // One pipeline stage
      always_ff @(posedge clk_i or negedge rst_ni)
      begin
        if(~rst_ni)
          output_valid <= 1'b0;
        else if(clear_i)
          output_valid <= 1'b0;
        else if(enable_i)
          output_valid <= activation_valid_hs & weight_valid_hs;
      end
    end
  end
  else begin : nopipe_stage_gen
    // No pipeline stage
    assign output_valid = activation_valid_hs & weight_valid_hs;
  end


  ////////////////////////
  // OUTPUT ASSIGNMENTS //
  ////////////////////////

  assign popcount_o.valid = output_valid;
  assign popcount_o.strb  = '1;
  assign popcount_o.data  = reduction_bitcount;


  ////////////////
  // ASSERTIONS //
  ////////////////
  // Ensure Pipeline stage locations are legal
  `ifndef SYNTHESIS
    assert property (@(posedge clk_i) PIPE_STAGE_LOCATION_0 < $clog2(TP)+1)
      else $fatal("PIPE_STAGE_LOCATION_0 is >= $clog2(TP)+1 = %d", $clog2(TP)+1);
    assert property (@(posedge clk_i) PIPE_STAGE_LOCATION_1 < $clog2(TP)+1)
      else $fatal("PIPE_STAGE_LOCATION_1 is >= $clog2(TP)+1 = %d", $clog2(TP)+1);
    assert property (@(posedge clk_i) PIPE_STAGE_LOCATION_2 < $clog2(TP)+1)
      else $fatal("PIPE_STAGE_LOCATION_2 is >= $clog2(TP)+1 = %d", $clog2(TP)+1);
  `endif

endmodule // rbe_binconv_sop

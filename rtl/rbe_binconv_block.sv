/*
 * rbe_binconv_block.sv
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
 * Each BinConv block takes as input a binary `TP`-vector of weights and
 * `BC_BLOCK_SIZE` binary `TP`-vectors of activations.
 * By default, `BC_BLOCK_SIZE`=4 and `TP`=32.
 *
 */


module rbe_binconv_block #(
  parameter  int unsigned BC_BLOCK_SIZE = rbe_package::BINCONV_BLOCK_SIZE, // number of SoP's per BinConv block (default 4)
  parameter  int unsigned TP            = rbe_package::BINCONV_TP,         // number of input elements processed per cycle
  parameter  int unsigned ROW_IDX       = 0,
  parameter  int unsigned PIPELINE      = 1,
  localparam int unsigned POPCOUNT_SIZE = TP,
  localparam int unsigned MAX_SHIFT     = rbe_package::MAX_SHIFT
) (
  // global signals
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              test_mode_i,
  // local enable & clear
  input  logic                              enable_i,
  input  logic                              clear_i,
  // input activation stream + handshake
  hwpe_stream_intf_stream.sink              activation_i [BC_BLOCK_SIZE-1:0],
  // input weight stream + handshake
  hwpe_stream_intf_stream.sink              weight_i,
  // output features + handshake
  hwpe_stream_intf_stream.source            block_pres_o,
  // control and flags
  input  rbe_package::ctrl_binconv_block_t  ctrl_i,
  output rbe_package::flags_binconv_block_t flags_o
);

  ///////////////////////////////////////
  // INTERFACE AND SIGNAL DECLARATIONS //
  ///////////////////////////////////////

  // internal weight interface
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP )
  ) weight_int [BC_BLOCK_SIZE-1:0] (.clk (clk_i));

  // BinConv result interface
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( POPCOUNT_SIZE )
  ) popcount   [BC_BLOCK_SIZE-1:0] (.clk (clk_i));
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( POPCOUNT_SIZE )
  ) popcount_q [BC_BLOCK_SIZE-1:0] (.clk (clk_i));

  // scaled BinConv result interface
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( POPCOUNT_SIZE+8 )
  ) popcount_scaled [BC_BLOCK_SIZE-1:0] (.clk (clk_i));

  // signals
  logic [POPCOUNT_SIZE+8+$clog2(BC_BLOCK_SIZE)-1:0] binconv_block_pres_d;
  logic [POPCOUNT_SIZE+8+$clog2(BC_BLOCK_SIZE)-1:0] binconv_block_pres_q;
  logic                                             binconv_block_pres_valid_d;
  logic                                             binconv_block_pres_valid_q;

  logic [BC_BLOCK_SIZE-1:0][POPCOUNT_SIZE-1:0] popcount_scaled_data;

  logic [$clog2(8):0] cnt_q, cnt_d;
  logic [$clog2(8):0] scale_fs_3;
  logic [      8-1:0] scale_fs_1;

  logic [BC_BLOCK_SIZE-1:0][$clog2(MAX_SHIFT):0] shift_sel_pre;

  logic [BC_BLOCK_SIZE-1:0] [POPCOUNT_SIZE-1:0]     popcount_data_q;
  logic [BC_BLOCK_SIZE-1:0] [(POPCOUNT_SIZE/8)-1:0] popcount_strb_q;
  logic [BC_BLOCK_SIZE-1:0] popcount_valid_q;
  logic [BC_BLOCK_SIZE-1:0] popcount_ready_q;

  rbe_package::ctrl_scale_t scale_ctrl   [BC_BLOCK_SIZE-1:0];
  rbe_package::ctrl_scale_t scale_ctrl_q [BC_BLOCK_SIZE-1:0];

  ///////////////////
  // SCALE CONTROL //
  ///////////////////

  // default shift for FS==1
  assign scale_fs_1 = 8'(ROW_IDX) & {8{(ctrl_i.row_onehot_en[ROW_IDX])}};

  // default shift for FS==3
  // count up to QW-1
  assign cnt_d      = (cnt_q == ctrl_i.qw-1) ? '0 : cnt_q + 1;
  assign scale_fs_3 = cnt_q;


  ///////////////////////////////
  // BINCONV AND SCALE MODULES //
  ///////////////////////////////

  // iterate over all BC_BLOCK_SIZE BinConvs in a singe block
  for(genvar ii=0; ii<BC_BLOCK_SIZE; ii+=1) begin : sop_gen

    assign weight_int[ii].data  = weight_i.data;
    assign weight_int[ii].valid = weight_i.valid;
    assign weight_int[ii].strb  = weight_i.strb;

    // TODO: activate / deactivate via control mask
    rbe_binconv_sop #(
      .TP ( TP )
    ) i_binconv_sop (
      .clk_i        ( clk_i             ),
      .rst_ni       ( rst_ni            ),
      .test_mode_i  ( test_mode_i       ),
      .enable_i     ( enable_i          ),
      .clear_i      ( clear_i           ),
      .activation_i ( activation_i [ii] ),
      .weight_i     ( weight_int [ii]   ),
      .popcount_o   ( popcount [ii]     ),
      .ctrl_i       ( ctrl_i.ctrl_sop   )
    );

    always_comb
    begin : shift_sel_comb
      // select the pre shift
      if (ctrl_i.offset_en == '0) begin
        if (ctrl_i.fs == 1) begin
          shift_sel_pre[ii] <= ii + scale_fs_1;
        end
        else begin
          shift_sel_pre[ii] <= ii + scale_fs_3;
        end
      end
      else begin
        if (ctrl_i.offset_state == 0) begin
        // first cycle offset computation: offset shift
          shift_sel_pre[ii] <= ii + ctrl_i.qw - 1;
        end
        else begin
        // second cycle offset computation: identity shift
          shift_sel_pre[ii] <= ii;
        end
      end
    end // shift_sel_comb

    // select lower four bit of activation bits or upper four bits
    assign scale_ctrl[ii].shift_sel = ctrl_i.qa_tile_sel ?
                                      shift_sel_pre[ii]+4 :
                                      shift_sel_pre[ii];

    // Pipeline the datapath if needed
    if (PIPELINE ==1 ) begin : pipe_stage_gen
      // add a pipieline stage
      always_ff @(posedge clk_i or negedge rst_ni)
      begin
        if(~rst_ni) begin
          popcount_data_q[ii]        <= '0;
          popcount_valid_q[ii]       <= '0;
          popcount_strb_q[ii]        <= '0;
          scale_ctrl_q[ii].shift_sel <= '0;
        end
        else if(clear_i) begin
          popcount_data_q[ii]        <= '0;
          popcount_valid_q[ii]       <= '0;
          popcount_strb_q[ii]        <= '0;
          scale_ctrl_q[ii].shift_sel <= '0;
        end
        else if(enable_i) begin
          popcount_data_q[ii]        <= popcount[ii].data;
          popcount_valid_q[ii]       <= popcount[ii].valid & popcount_q[ii].ready;
          popcount_strb_q[ii]        <= popcount[ii].strb;
          scale_ctrl_q[ii].shift_sel <= scale_ctrl[ii].shift_sel;
        end
      end
      assign popcount_q[ii].data  = popcount_data_q [ii];
      assign popcount_q[ii].valid = popcount_valid_q [ii];
      assign popcount_q[ii].strb  = popcount_strb_q [ii];
      assign popcount[ii].ready   = popcount_q[ii].ready;
    end
    else begin
      // assign the streams directly
      hwpe_stream_assign i_assign_popcount(
        .push_i(popcount[ii]),
        .pop_o(popcount_q[ii])
      );
      assign scale_ctrl_q[ii].shift_sel = scale_ctrl[ii].shift_sel;
    end

    // shift the results by the correct number of bits
    rbe_scale #(
      .INP_ACC     ( POPCOUNT_SIZE     ),
      .OUT_ACC     ( POPCOUNT_SIZE + 8 ),
      .N_SHIFTS    ( 8+8               )
    ) i_binconv_scale (
      .clk_i       ( clk_i                      ),
      .rst_ni      ( rst_ni                     ),
      .test_mode_i ( test_mode_i                ),
      .data_i      ( popcount_q [ii]            ),
      .data_o      ( popcount_scaled [ii]       ),
      .ctrl_i      ( scale_ctrl_q[ii].shift_sel ),
      .flags_o     ( flags_o.flags_scale[ii]    )
    );

    assign popcount_scaled_data[ii] = popcount_scaled[ii].data;

  end // sop_gen

  // sum up all four convoluted and shifted results
  always_comb
  begin
    binconv_block_pres_d = '0;
    for(int i=0; i<BC_BLOCK_SIZE; i++) begin
      binconv_block_pres_d += popcount_scaled_data[i];
    end
  end

  assign binconv_block_pres_valid_d = popcount_scaled[0].valid;


  ///////////////
  // REGISTERS //
  ///////////////

  // registers for block results
  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        binconv_block_pres_q <= '0;
      else if(clear_i)
        binconv_block_pres_q <= '0;
      else if(popcount_scaled[0].valid & popcount_scaled[0].ready)
        binconv_block_pres_q <= binconv_block_pres_d;
    end

  // registers for output valid signal
  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        binconv_block_pres_valid_q <= '0;
      else if(clear_i)
        binconv_block_pres_valid_q <= '0;
      else if(popcount_scaled[0].ready)
        binconv_block_pres_valid_q <= binconv_block_pres_valid_d;
    end

  // registers for shift counter
  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni)
      cnt_q <= '0;
    else if(popcount[0].valid & popcount[0].ready & (~ctrl_i.offset_en))
      cnt_q <= cnt_d;
  end


  ////////////////////////
  // OUTPUT ASSIGNMENTS //
  ////////////////////////

  assign weight_i.ready = weight_int[0].ready;

  assign block_pres_o.valid = binconv_block_pres_valid_q;
  assign block_pres_o.data  = binconv_block_pres_q;

  for(genvar ii=0; ii<BC_BLOCK_SIZE; ii++) begin : ready_prop_gen
    assign popcount_scaled[ii].ready = block_pres_o.ready;
  end // ready_prop_gen

endmodule // rbe_binconv_block

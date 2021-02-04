/*
 * rbe_binconv_column.sv
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
 * TODO Description
 *
 */


module rbe_binconv_column #(
  parameter  int unsigned BC_COLUMN_SIZE   = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv blocks per column (default 9)
  parameter  int unsigned BC_BLOCK_SIZE    = rbe_package::BINCONV_BLOCK_SIZE,  // number of SoP's per BinConv block (default 4)
  parameter  int unsigned TP               = rbe_package::BINCONV_TP,          // number of input elements processed per cycle
  localparam int unsigned BC_COLBLOCK_SIZE = BC_COLUMN_SIZE*BC_BLOCK_SIZE,
  localparam int unsigned POPCOUNT_SIZE    = TP,
  localparam int unsigned BLOCK_PRES_SIZE  = POPCOUNT_SIZE+8+$clog2(BC_BLOCK_SIZE),
  localparam int unsigned COLUMN_PRES_SIZE = BLOCK_PRES_SIZE+$clog2(BC_COLUMN_SIZE)
) (
  // global signals
  input  logic                               clk_i,
  input  logic                               rst_ni,
  input  logic                               test_mode_i,
  // local enable & clear
  input  logic                               enable_i,
  input  logic                               clear_i,
  // input activation stream + handshake
  hwpe_stream_intf_stream.sink               activation_i  [BC_COLBLOCK_SIZE-1:0],
  // input weight stream + handshake
  hwpe_stream_intf_stream.sink               weight_i      [  BC_COLUMN_SIZE-1:0],
  // output features + handshake
  hwpe_stream_intf_stream.source             column_pres_o,
  // control and flags
  input  rbe_package::ctrl_binconv_column_t  ctrl_i,
  output rbe_package::flags_binconv_column_t flags_o
);

  ////////////////////////////
  // INTERFACES AND SIGNALS //
  ////////////////////////////

`ifndef SYNTHESIS
  hwpe_stream_intf_stream #(
    .DATA_WIDTH        ( BLOCK_PRES_SIZE ),
    .BYPASS_VCR_ASSERT ( 1'b1            ),
    .BYPASS_VDR_ASSERT ( 1'b1            )
  ) block_pres [BC_COLUMN_SIZE-1:0] (.clk(clk_i));
`else
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( BLOCK_PRES_SIZE )
  ) block_pres [BC_COLUMN_SIZE-1:0] (.clk(clk_i));
`endif

  logic [COLUMN_PRES_SIZE-1:0]   binconv_column_pres_d, binconv_column_pres_q;
  logic                          binconv_column_pres_valid_d, binconv_column_pres_valid_q;
  logic [COLUMN_PRES_SIZE/8-1:0] binconv_column_pres_strb_d, binconv_column_pres_strb_q;

  logic [BC_COLUMN_SIZE-1:0][BLOCK_PRES_SIZE-1:0] block_pres_data;


  ///////////////////
  // BLOCK MODULES //
  ///////////////////

  for(genvar ii=0; ii<BC_COLUMN_SIZE; ii++) begin : block_gen

    rbe_binconv_block #(
      .BC_BLOCK_SIZE ( BC_BLOCK_SIZE ),
      .TP            ( TP            ),
      .ROW_IDX       ( ii            )
    ) i_block (
      .clk_i        ( clk_i                                                  ),
      .rst_ni       ( rst_ni                                                 ),
      .test_mode_i  ( test_mode_i                                            ),
      .enable_i     ( ctrl_i.ctrl_block.row_onehot_en[ii]                    ),
      .clear_i      ( clear_i                                                ),
      .activation_i ( activation_i [(ii+1)*BC_BLOCK_SIZE-1:ii*BC_BLOCK_SIZE] ),
      .weight_i     ( weight_i [ii]                                          ),
      .block_pres_o ( block_pres [ii]                                        ),
      .ctrl_i       ( ctrl_i.ctrl_block                                      ),
      .flags_o      ( flags_o.flags_block[ii]                                )
    );

    assign block_pres_data[ii] = block_pres[ii].data;

  end // block_gen


  ///////////////////////////////////
  // COMPUTATION OF COLUMN RESULTS //
  ///////////////////////////////////
  always_comb
  begin
    binconv_column_pres_d = '0;
    for(int i=0; i<BC_COLUMN_SIZE; i++) begin
      binconv_column_pres_d += block_pres_data[i];
    end
  end
  assign binconv_column_pres_valid_d = block_pres[0].valid;
  assign binconv_column_pres_strb_d  = block_pres[0].strb;


  ////////////////////////
  // OUTPUT ASSIGNMENTS //
  ////////////////////////
  assign column_pres_o.valid = binconv_column_pres_valid_q;
  assign column_pres_o.strb  = binconv_column_pres_strb_q;
  assign column_pres_o.data  = binconv_column_pres_q;

  for(genvar ii=0; ii<BC_COLUMN_SIZE; ii++) begin : ready_prop_gen
    assign block_pres[ii].ready = column_pres_o.ready;
  end // ready_prop_gen


  ///////////////
  // REGISTERS //
  ///////////////
  // registers for column results
  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni)
      binconv_column_pres_q <= '0;
    else if(clear_i)
      binconv_column_pres_q <= '0;
    else if(block_pres[0].valid & block_pres[0].ready)
      binconv_column_pres_q <= binconv_column_pres_d;
  end

  // registers for output valid signal
  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      binconv_column_pres_valid_q <= '0;
      binconv_column_pres_strb_q  <= '0;
    end
    else if(clear_i) begin
      binconv_column_pres_valid_q <= '0;
      binconv_column_pres_strb_q  <= '0;
    end
    else if(block_pres[0].ready) begin
      binconv_column_pres_valid_q <= binconv_column_pres_valid_d;
      binconv_column_pres_strb_q  <= binconv_column_pres_strb_d;
    end
  end

endmodule // rbe_binconv_column

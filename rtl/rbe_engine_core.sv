/*
 * rbe_engine_core.sv
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


module rbe_engine_core #(
  parameter  int unsigned RBE_LD_SIZE       = 9,
  parameter  int unsigned BC_COLUMN_SIZE    = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv blocks per column (default 9)
  parameter  int unsigned BC_NR_COLUMN      = rbe_package::BINCONV_NR_COLUMN,   // number of BinConv columns (default 9 -- same of size of BinConv columns!)
  parameter  int unsigned BC_BLOCK_SIZE     = rbe_package::BINCONV_BLOCK_SIZE,  // number of SoP's per BinConv block (default 4)
  parameter  int unsigned BC_NR_ACTIVATIONS = 25*BC_BLOCK_SIZE,                 // TODO FIXME
  parameter  int unsigned TP                = rbe_package::BINCONV_TP,          // number of input elements processed per cycle
  localparam int unsigned BW                = rbe_package::BANDWIDTH,
  localparam int unsigned POPCOUNT_SIZE     = $clog2(TP)+1,
  localparam int unsigned BLOCK_PRES_SIZE   = POPCOUNT_SIZE+8+$clog2(BC_BLOCK_SIZE),
  localparam int unsigned COLUMN_PRES_SIZE  = BLOCK_PRES_SIZE+$clog2(BC_COLUMN_SIZE)+$clog2(BC_COLUMN_SIZE)
) (
  // global signals
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            test_mode_i,
  // local enable & clear
  input  logic                            enable_i,
  input  logic                            clear_i,
  // input streams + handshake
  hwpe_stream_intf_stream.sink            activation_flatten_i [BC_NR_ACTIVATIONS-1:0],
  hwpe_stream_intf_stream.sink            weight_i             [      RBE_LD_SIZE-1:0],
  hwpe_stream_intf_stream.sink            norm_i,
  hwpe_stream_intf_stream.source          conv_o               [     BC_NR_COLUMN-1:0],
  // control and flags
  input  rbe_package::ctrl_engine_core_t  ctrl_i,
  output rbe_package::flags_engine_core_t flags_o
);

  ///////////////////////////////////////
  // INTERFACE AND SIGNAL DECLARATIONS //
  ///////////////////////////////////////

  logic                                          all_norm_ready;
  logic [$clog2(BC_NR_COLUMN):0][BC_NR_COLUMN:0] all_norm_ready_tree;

  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP )
  ) activation [BC_NR_ACTIVATIONS-1:0] (.clk(clk_i));

  // assert stream handshakes only when not doing synthesis
`ifndef SYNTHESIS
  hwpe_stream_intf_stream #(
    .DATA_WIDTH       ( COLUMN_PRES_SIZE ),
    .BYPASS_VCR_ASSERT( 1'b1             ),
    .BYPASS_VDR_ASSERT( 1'b1             )
  ) pres [BC_NR_COLUMN-1:0] (.clk(clk_i));
`else
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( COLUMN_PRES_SIZE )
  ) pres [BC_NR_COLUMN-1:0] (.clk(clk_i));
`endif

`ifndef SYNTHESIS
  hwpe_stream_intf_stream #(
    .DATA_WIDTH       ( BW   ),
    .BYPASS_VCR_ASSERT( 1'b1 ),
    .BYPASS_VDR_ASSERT( 1'b1 )
  ) norm [BC_NR_COLUMN-1:0] (.clk(clk_i));
`else
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( BW )
  ) norm [BC_NR_COLUMN-1:0] (.clk(clk_i));
`endif


  ////////////////////////////////////////////
  // NORMALIZATION READY SIGNAL COMBINATION //
  ////////////////////////////////////////////

  for(genvar i=0; i<BC_NR_COLUMN; i++) begin
    assign all_norm_ready_tree[0][i] = norm[i].ready;
    assign norm[i].data  = norm_i.data;
    assign norm[i].valid = norm_i.valid;
    assign norm[i].strb  = norm_i.strb;
  end

  for(genvar j=0; j<$clog2(BC_NR_COLUMN); j++) begin
    for(genvar i=0; i<BC_NR_COLUMN; i+=2) begin
      assign all_norm_ready_tree[j+1][i/2] = all_norm_ready_tree[j][i] & all_norm_ready_tree[j][i+1];
    end
  end

  assign all_norm_ready = all_norm_ready_tree[($clog2(BC_NR_COLUMN)-1)][0];
  assign norm_i.ready   = all_norm_ready;


  /////////////////////
  // INPUT REGISTERS //
  /////////////////////
  for (genvar ii=0; ii<BC_NR_ACTIVATIONS; ii++) begin : activation_reg_gen
    rbe_input_register #(
      .TP ( TP )
    ) i_activation_reg (
      .clk_i            ( clk_i                            ),
      .rst_ni           ( rst_ni                           ),
      .test_mode_i      ( test_mode_i                      ),
      .enable_i         ( enable_i                         ),
      .clear_i          ( clear_i                          ),
      .ctrl_feat_buf_i  ( ctrl_i.ctrl_activation_reg       ),
      .flags_feat_buf_o ( flags_o.flags_activation_reg[ii] ),
      .feat_i           ( activation_flatten_i[ii]         ),
      .feat_o           ( activation[ii]                   )
    );
  end // activation_reg_gen


  ///////////////////
  // BINCONV ARRAY //
  ///////////////////
  rbe_binconv_array #(
    .BC_COLUMN_SIZE    ( BC_COLUMN_SIZE    ),
    .BC_NR_COLUMN      ( BC_NR_COLUMN      ),
    .BC_BLOCK_SIZE     ( BC_BLOCK_SIZE     ),
    .BC_NR_ACTIVATIONS ( BC_NR_ACTIVATIONS ),
    .TP                ( TP                )
  ) i_binconv_array (
    .clk_i        ( clk_i                       ),
    .rst_ni       ( rst_ni                      ),
    .test_mode_i  ( test_mode_i                 ),
    .enable_i     ( enable_i                    ),
    .clear_i      ( clear_i                     ),
    .activation_i ( activation                  ),
    .weight_i     ( weight_i                    ),
    .pres_o       ( pres                        ),
    .ctrl_i       ( ctrl_i.ctrl_array           ),
    .flags_o      ( flags_o.flags_binconv_array )
  );


  /////////////////////////////////////////////////
  // ACCUMULATORS + NORMALIZATION / QUANTIZATION //
  /////////////////////////////////////////////////

  for (genvar ii=0; ii<BC_NR_COLUMN; ii++) begin : accumulator_gen
    rbe_accumulator_normquant #(
      .TP  ( TP ),
      .AP  ( TP ),
      .ACC ( 32 )
    ) i_accumulator (
      .clk_i       ( clk_i                ),
      .rst_ni      ( rst_ni               ),
      .test_mode_i ( test_mode_i          ),
      .enable_i    ( enable_i             ),
      .clear_i     ( clear_i              ),
      .conv_i      ( pres[ii]             ),
      .norm_i      ( norm[ii]             ),
      .conv_o      ( conv_o[ii]           ),
      .ctrl_i      ( ctrl_i.ctrl_aq       ),
      .flags_o     ( flags_o.flags_aq[ii] )
    );
  end // accumulator_gen

endmodule // rbe_engine_core

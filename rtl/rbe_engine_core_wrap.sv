/*
 * rbe_engine_core_wrap.sv
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


module rbe_engine_core_wrap
  import rbe_package::*;
#(
  parameter int unsigned RBE_LD_SIZE       = 9,
  parameter int unsigned BC_COLUMN_SIZE    = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv blocks per column (default 9)
  parameter int unsigned BC_NR_COLUMN      = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv columns (default 9 -- same of size of BinConv columns!)
  parameter int unsigned BC_BLOCK_SIZE     = rbe_package::BINCONV_BLOCK_SIZE,  // number of SoP's per BinConv block (default 4)
  parameter int unsigned BC_NR_ACTIVATIONS = 25*4,                             // TODO FIXME
  parameter int unsigned TP_IN             = rbe_package::BINCONV_TP           // number of input elements processed per cycle
) (
  // global signals
  input  logic                                     clk_i,
  input  logic                                     rst_ni,
  input  logic                                     test_mode_i,
  // local enable & clear
  input  logic                                     enable_i,
  input  logic                                     clear_i,
  // input streams + handshake
  input  logic [BC_NR_ACTIVATIONS-1:0]              load_activation_demuxed_valid,
  input  logic [BC_NR_ACTIVATIONS-1:0][TP_IN-1:0]   load_activation_demuxed_data,
  input  logic [BC_NR_ACTIVATIONS-1:0][TP_IN/8-1:0] load_activation_demuxed_strb,
  output logic [BC_NR_ACTIVATIONS-1:0]              load_activation_demuxed_ready,
  input  logic [RBE_LD_SIZE-1:0]                    load_weight_valid,
  input  logic [RBE_LD_SIZE-1:0][TP_IN-1:0]         load_weight_data,
  input  logic [RBE_LD_SIZE-1:0][TP_IN/8-1:0]       load_weight_strb,
  output logic [RBE_LD_SIZE-1:0]                    load_weight_ready,
  output logic [BC_NR_COLUMN-1:0]                   store_out_valid,
  output logic [BC_NR_COLUMN-1:0][TP_IN-1:0]        store_out_data,
  output logic [BC_NR_COLUMN-1:0][TP_IN/8-1:0]      store_out_strb,
  input  logic [BC_NR_COLUMN-1:0]                   store_out_ready,
  input  ctrl_engine_core_t                         ctrl_i,
  output flags_engine_core_t                        flags_o
);

  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP_IN )
  ) load_activation_demuxed [BC_NR_ACTIVATIONS-1:0] (
    .clk ( clk_i )
  );
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP_IN )
  ) load_weight             [RBE_LD_SIZE-1:0] (
    .clk ( clk_i )
  );

  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP_IN )
  ) store_out               [BC_NR_COLUMN-1:0] (
    .clk ( clk_i )
  );

  generate;
    for (genvar ii=0; ii<BC_NR_ACTIVATIONS; ii++) begin
      assign load_activation_demuxed[ii].valid   = load_activation_demuxed_valid [ii];
      assign load_activation_demuxed[ii].data    = load_activation_demuxed_data [ii];
      assign load_activation_demuxed[ii].strb    = load_activation_demuxed_strb [ii];
      assign load_activation_demuxed_ready [ii]  = load_activation_demuxed[ii].ready;
    end
    for (genvar ii=0; ii<RBE_LD_SIZE; ii++) begin
      assign load_weight[ii].valid   = load_weight_valid [ii];
      assign load_weight[ii].data    = load_weight_data [ii];
      assign load_weight[ii].strb    = load_weight_strb [ii];
      assign load_weight_ready [ii]  = load_weight[ii].ready;
    end
    for (genvar ii=0; ii<BC_NR_COLUMN; ii++) begin
      assign store_out_valid [ii] = store_out[ii].valid;
      assign store_out_data  [ii] = store_out[ii].data;
      assign store_out_strb  [ii] = store_out[ii].strb;
      assign store_out[ii].ready  = store_out_ready [ii];
    end
  endgenerate

  rbe_engine_core #(
    .RBE_LD_SIZE       ( RBE_LD_SIZE       ),
    .BC_COLUMN_SIZE    ( BC_COLUMN_SIZE    ),
    .BC_NR_COLUMN      ( BC_NR_COLUMN      ),
    .BC_BLOCK_SIZE     ( BC_BLOCK_SIZE     ),
    .BC_NR_ACTIVATIONS ( BC_NR_ACTIVATIONS ),
    .TP_IN             ( TP_IN             )
  ) i_engine_core (
    .clk_i                   ( clk_i                   ),
    .rst_ni                  ( rst_ni                  ),
    .test_mode_i             ( test_mode_i             ),
    .enable_i                ( enable_i                ),
    .clear_i                 ( clear_i                 ),
    .load_activation_demuxed ( load_activation_demuxed ),
    .load_weight             ( load_weight             ),
    .store_out               ( store_out               ),
    .ctrl_i                  ( ctrl_i                  ),
    .flags_o                 ( flags_o                 )
  );

endmodule // rbe_engine_core_wrap

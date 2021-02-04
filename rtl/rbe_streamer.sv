/*
 * rbe_streamer.sv
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
 * Streamer Unit of the RBE
 *
 */

module rbe_streamer #(
  parameter int unsigned TP  = rbe_package::BITS_PER_TCDM_PORT,
  parameter int unsigned MP  = rbe_package::NR_TCDM_PORTS,
  localparam int unsigned BW = TP * MP
) (
  // global signals
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         test_mode_i,
  // local enable & clear
  input  logic                         enable_i,
  input  logic                         clear_i,
  // input feat stream + handshake
  hwpe_stream_intf_stream.source       feat_o,
  // input weight stream + handshake
  hwpe_stream_intf_stream.source       weight_o,
  // input norm stream + handshake
  hwpe_stream_intf_stream.source       norm_o,
  // output features + handshake
  hwpe_stream_intf_stream.sink         conv_i,
  // TCDM ports
  hci_core_intf.master                 tcdm,
  // control channel
  input  rbe_package::ctrl_streamer_t  ctrl_i,
  output rbe_package::flags_streamer_t flags_o
);


/* Streamer dataflow layout
 *
 *            i_tcdm_fifo      /| <--- sink   <------------ conv_i
 *            -----------     | |
 *   tcdm --- |    |    | --- | |                   /| ---> feat_o
 *            -----------     | |                  | |
 *              depth 2        \| ---> source ---> | | ---> weight_o
 *                                                 | |
 *                                                  \| ---> norm_o
 */

  // control and flag structs
  hci_package::hci_streamer_ctrl_t  all_source_ctrl;
  hci_package::hci_streamer_flags_t all_source_flags;
  hwpe_stream_package::flags_fifo_t tcdm_fifo_flags;

  // stream interfaces
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) all_source       (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) virt_source[2:0] (.clk(clk_i));

  // hci-tcdm interfaces
  hci_core_intf #(.DW (BW)) virt_tcdm [1:0] (.clk(clk_i));
  hci_core_intf #(.DW (BW)) tcdm_prefifo    (.clk(clk_i));


  // select correct source for source ctrl and data stream
  logic [1:0] ld_which_mux_sel;

  always_comb
  begin : ld_which_ctrl_mux
    // default
    all_source_ctrl  = '0;
    ld_which_mux_sel = 2'b00;
    // load features
    if(ctrl_i.ld_which_mux_sel == rbe_package::LD_FEAT_SEL) begin
      all_source_ctrl  = ctrl_i.feat_source_ctrl;
      ld_which_mux_sel = 2'b00;
    end
    // load weights
    else if(ctrl_i.ld_which_mux_sel == rbe_package::LD_WEIGHT_SEL) begin
      all_source_ctrl  = ctrl_i.weight_source_ctrl;
      ld_which_mux_sel = 2'b01;
    end
    // load normalization params
    else if(ctrl_i.ld_which_mux_sel == rbe_package::LD_NORM_SEL) begin
      all_source_ctrl  = ctrl_i.norm_source_ctrl;
      ld_which_mux_sel = 2'b10;
    end
  end

  // source address generation etc.
  hci_core_source #(
    .DATA_WIDTH ( BW )
  ) i_all_source (
    .clk_i       ( clk_i            ),
    .rst_ni      ( rst_ni           ),
    .test_mode_i ( test_mode_i      ),
    .clear_i     ( clear_i          ),
    .tcdm        ( virt_tcdm [0]    ),
    .stream      ( all_source       ),
    .ctrl_i      ( all_source_ctrl  ),
    .flags_o     ( all_source_flags )
  );
  // assign source flags
  assign flags_o.feat_source_flags   = all_source_flags;
  assign flags_o.norm_source_flags   = all_source_flags;
  assign flags_o.weight_source_flags = all_source_flags;

  // demultiplex the incoming source data according to the data which is loaded:
  // features, weights, or normalization parameters
  hwpe_stream_demux_static #(
    .NB_OUT_STREAMS ( 3 )
  ) i_all_source_demux (
    .clk_i   ( clk_i            ),
    .rst_ni  ( rst_ni           ),
    .clear_i ( clear_i          ),
    .sel_i   ( ld_which_mux_sel ),
    .push_i  ( all_source       ),
    .pop_o   ( virt_source      )
  );

  hwpe_stream_assign i_assign_feat   ( .push_i (virt_source[0]), .pop_o (feat_o) );
  hwpe_stream_assign i_assign_weight ( .push_i (virt_source[1]), .pop_o (weight_o) );
  hwpe_stream_assign i_assign_norm   ( .push_i (virt_source[2]), .pop_o (norm_o) );


  // sink address generation etc.
  hci_core_sink #(
    .DATA_WIDTH ( BW )
  ) i_sink (
    .clk_i       ( clk_i                   ),
    .rst_ni      ( rst_ni                  ),
    .test_mode_i ( test_mode_i             ),
    .clear_i     ( clear_i                 ),
    .tcdm        ( virt_tcdm [1]           ),
    .stream      ( conv_i                  ),
    .ctrl_i      ( ctrl_i.conv_sink_ctrl   ),
    .flags_o     ( flags_o.conv_sink_flags )
  );

  // either store or load data back into the memory
  hci_core_mux_static #(
    .NB_CHAN ( 2  ),
    .DW      ( BW )
  ) i_ld_st_mux_static (
    .clk_i   ( clk_i                ),
    .rst_ni  ( rst_ni               ),
    .clear_i ( clear_i              ),
    .sel_i   ( ctrl_i.ld_st_mux_sel ),
    .in      ( virt_tcdm            ),
    .out     ( tcdm_prefifo         )
  );

  // FIFO inbetween TCDM and SINK/SOURCE
  hci_core_fifo #(
    .FIFO_DEPTH ( 2  ),
    .DW         ( BW ),
    .AW         ( TP ),
    .OW         ( 1  )
  ) i_tcdm_fifo (
    .clk_i       ( clk_i           ),
    .rst_ni      ( rst_ni          ),
    .clear_i     ( clear_i         ),
    .flags_o     ( tcdm_fifo_flags ),
    .tcdm_slave  ( tcdm_prefifo    ),
    .tcdm_master ( tcdm            )
  );
  // assign fifo flags
  assign flags_o.tcdm_fifo_empty     = tcdm_fifo_flags.empty;

endmodule // rbe_streamer

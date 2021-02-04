/*
 * rbe_top.sv
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
 * Top-level RBE module instantiating the streamer, engine and control unit.
 *
 */


module rbe_top #(
  parameter  int unsigned ID_WIDTH       = rbe_package::ID_WIDTH,
  parameter  int unsigned N_CORES        = rbe_package::NR_CORES,
  parameter  int unsigned N_CONTEXT      = rbe_package::NR_CONTEXT,
  parameter  int unsigned BW             = rbe_package::BANDWIDTH,
  localparam int unsigned TP             = rbe_package::BINCONV_TP,
  localparam int unsigned MP             = BW/32, // number of memory ports (each a 32bit data)
  localparam int unsigned N_IO_REGS      = rbe_package::NR_IO_REGS,
  localparam int unsigned N_GENERIC_REGS = rbe_package::NR_GENERIC_REGS,
  localparam int unsigned REGFILE_N_EVT  = hwpe_ctrl_package::REGFILE_N_EVT
) (
  // global signals
  input  logic                                  clk_i,
  input  logic                                  rst_ni,
  input  logic                                  test_mode_i,
  // hci interconnect control
  output hci_package::hci_interconnect_ctrl_t   hci_ctrl_o,
  // events
  output logic [N_CORES-1:0][REGFILE_N_EVT-1:0] evt_o,
  // tcdm master ports
  hci_core_intf.master                          tcdm,
  // periph slave port
  hwpe_ctrl_intf_periph.slave                   periph
);

  ///////////////////////////////////////
  // INTERFACE AND SIGNAL DECLARATIONS //
  ///////////////////////////////////////

  // signals
  logic enable;
  logic clear;

  // streamer flags and control structs
  rbe_package::ctrl_streamer_t  streamer_ctrl;
  rbe_package::flags_streamer_t streamer_flags;
  rbe_package::ctrl_engine_t    engine_ctrl;
  rbe_package::flags_engine_t   engine_flags;

  // data streams
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) feat   (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) weight (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) norm   (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) conv   (.clk(clk_i));

  assign enable = 1'b1;


  /////////////////////////////
  // MAIN COMPUTATION ENGINE //
  /////////////////////////////

  rbe_engine #(
    .BW          ( BW )
  ) i_engine (
    .clk_i      (clk_i       ),
    .rst_ni     (rst_ni      ),
    .test_mode_i(test_mode_i ),
    .enable_i   (enable      ),
    .clear_i    (clear       ),
    .feat_i     (feat        ),
    .weight_i   (weight      ),
    .norm_i     (norm        ),
    .conv_o     (conv        ),
    .ctrl_i     (engine_ctrl ),
    .flags_o    (engine_flags)
  );


  //////////////
  // STREAMER //
  //////////////

  rbe_streamer #(
    .TP ( TP ),
    .MP ( MP )
  ) i_streamer (
    .clk_i      (clk_i         ),
    .rst_ni     (rst_ni        ),
    .test_mode_i(test_mode_i   ),
    .enable_i   (enable        ),
    .clear_i    (clear         ),
    .feat_o     (feat          ),
    .weight_o   (weight        ),
    .norm_o     (norm          ),
    .conv_i     (conv          ),
    .tcdm       (tcdm          ),
    .ctrl_i     (streamer_ctrl ),
    .flags_o    (streamer_flags)
  );


  /////////////
  // CONTROL //
  /////////////

  rbe_ctrl #(
    .ID_WIDTH       ( ID_WIDTH       ),
    .N_IO_REGS      ( N_IO_REGS      ),
    .N_CORES        ( N_CORES        ),
    .N_GENERIC_REGS ( N_GENERIC_REGS ),
    .N_CONTEXT      ( N_CONTEXT      )
) i_ctrl (
    .clk_i           (clk_i         ),
    .rst_ni          (rst_ni        ),
    .test_mode_i     (test_mode_i   ),
    .ctrl_hci_o      (hci_ctrl_o    ),
    .evt_o           (evt_o         ),
    .clear_o         (clear         ),
    .ctrl_streamer_o (streamer_ctrl ),
    .flags_streamer_i(streamer_flags),
    .ctrl_engine_o   (engine_ctrl   ),
    .flags_engine_i  (engine_flags  ),
    .periph          (periph        )
  );

endmodule // rbe_top

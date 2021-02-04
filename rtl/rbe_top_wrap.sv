/*
 * rbe_top_wrap.sv
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
 * Wrapper for the top-level RBE module which flattens the interface signals.
 *
 */


module rbe_top_wrap #(
  parameter  int unsigned ID_WIDTH       = rbe_package::ID_WIDTH,
  parameter  int unsigned N_CORES        = rbe_package::NR_CORES,
  parameter  int unsigned N_CONTEXT      = rbe_package::NR_CONTEXT,
  parameter  int unsigned BW             = rbe_package::BANDWIDTH,
  localparam int unsigned TP             = rbe_package::BINCONV_TP,
  localparam int unsigned MP             = BW/32, // number of memory ports (each a 32-bit data)
  localparam int unsigned REGFILE_N_EVT  = hwpe_ctrl_package::REGFILE_N_EVT
) (
  // global signals
  input  logic                                  clk_i,
  input  logic                                  rst_ni,
  input  logic                                  test_mode_i,
  // hci interconnect control
  output hci_package::hci_interconnect_ctrl_t   hci_ctrl_o,
  // evnets
  output logic [N_CORES-1:0][REGFILE_N_EVT-1:0] evt_o,
  // tcdm master ports
  output logic [     MP-1:0]                    tcdm_req,
  input  logic [     MP-1:0]                    tcdm_gnt,
  output logic [     MP-1:0][             31:0] tcdm_add,
  output logic [     MP-1:0]                    tcdm_wen,
  output logic [     MP-1:0][              3:0] tcdm_be,
  output logic [     MP-1:0][             31:0] tcdm_data,
  input  logic [     MP-1:0][             31:0] tcdm_r_data,
  input  logic [     MP-1:0]                    tcdm_r_valid,
  // periph slave port
  input  logic                                  periph_req,
  output logic                                  periph_gnt,
  input  logic [        31:0]                   periph_add,
  input  logic                                  periph_wen,
  input  logic [         3:0]                   periph_be,
  input  logic [        31:0]                   periph_data,
  input  logic [ID_WIDTH-1:0]                   periph_id,
  output logic [        31:0]                   periph_r_data,
  output logic                                  periph_r_valid,
  output logic [ID_WIDTH-1:0]                   periph_r_id
);

  //////////////////////////////////////
  // INTERFACE AND SIGNAL DECLARATION //
  //////////////////////////////////////

  hci_core_intf #(.DW(BW)) tcdm (.clk(clk_i));

  hwpe_ctrl_intf_periph #(.ID_WIDTH(ID_WIDTH)) periph (.clk(clk_i));


  ///////////////////
  // TCDM BINDINGS //
  ///////////////////
  // Assign every TCDM master port a each 32-bit data
  for(genvar ii=0; ii<MP; ii++) begin: tcdm_binding
    assign tcdm_req  [ii] = tcdm.req;
    assign tcdm_add  [ii] = tcdm.add + ii*4;
    assign tcdm_wen  [ii] = tcdm.wen;
    assign tcdm_be   [ii] = tcdm.be[(ii+1)*4-1:ii*4];
    assign tcdm_data [ii] = tcdm.data[(ii+1)*32-1:ii*32];
  end

  assign tcdm.gnt     = &(tcdm_gnt);
  assign tcdm.r_valid = &(tcdm_r_valid);
  assign tcdm.r_data  = { >> {tcdm_r_data} };


  /////////////////////////
  // PERIPHERAL BINDINGS //
  /////////////////////////

  // Assign all Peripheral slave port signals
  always_comb
    begin
      periph.req     = periph_req;
      periph.add     = periph_add;
      periph.wen     = periph_wen;
      periph.be      = periph_be;
      periph.data    = periph_data;
      periph.id      = periph_id;
      periph_gnt     = periph.gnt;
      periph_r_data  = periph.r_data;
      periph_r_valid = periph.r_valid;
      periph_r_id    = periph.r_id;
    end


  ////////////////
  // RBE MODULE //
  ////////////////

  rbe_top #(
    .ID_WIDTH ( ID_WIDTH  ),
    .N_CORES  ( N_CORES   ),
    .N_CONTEXT( N_CONTEXT ),
    .BW       ( BW        )
  ) i_rbe_top (
    .clk_i       ( clk_i        ),
    .rst_ni      ( rst_ni       ),
    .test_mode_i ( test_mode_i  ),
    .hci_ctrl_o  ( hci_ctrl_o   ),
    .evt_o       ( evt_o        ),
    .tcdm        ( tcdm.master  ),
    .periph      ( periph.slave )
  );

endmodule // rbe_top_wrap

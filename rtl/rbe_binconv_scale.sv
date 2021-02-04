/*
 * rbe_binconv_scale.sv
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
 * Shifts incoming data `INP_ACC`-bit data_i by the amount up to `N_SHIFTS`
 * as selected by ctrl_i.shift_sel.
 * The result data_o has `OUT_ACC`-bits.
 *
 */


module rbe_scale #(
  parameter int unsigned INP_ACC  =  8, // input bitwidth
  parameter int unsigned OUT_ACC  = 16, // output bitwidth
  parameter int unsigned N_SHIFTS =  8  // number of mutliplexed shifts
) (
  // global signals
  input logic                       clk_i,
  input logic                       rst_ni,
  input logic                       test_mode_i,
  // input data
  hwpe_stream_intf_stream.sink      data_i,
  // output data
  hwpe_stream_intf_stream.source    data_o,
  // control and flags
  input  rbe_package::ctrl_scale_t  ctrl_i,
  output rbe_package::flags_scale_t flags_o
);

  /////////////
  // SIGNALS //
  /////////////

  logic [OUT_ACC-1:0] shifted_data  [N_SHIFTS-1:0];
  logic [OUT_ACC-1:0] unshifted_data;


  ///////////
  // SHIFT //
  ///////////

  // start with input data
  assign unshifted_data[INP_ACC-1:0] = data_i.data[INP_ACC-1:0];

  // fix assignment for all parameter combinations
  if (OUT_ACC-1 >= INP_ACC) begin
    assign unshifted_data[OUT_ACC-1:INP_ACC] = '0;
  end

  // shift
  for(genvar i=0; i<N_SHIFTS; i++) begin
    assign shifted_data[i] = unshifted_data << i;
  end

  ///////////////////////
  // OUTPUT ASSIGNMENT //
  ///////////////////////
  assign data_o.data  = shifted_data[ctrl_i.shift_sel];
  assign data_o.valid = data_i.valid;
  assign data_o.strb  = data_i.strb;

  assign data_i.ready = data_o.ready;

  assign flags_o.shift_sel = ctrl_i.shift_sel;

endmodule // rbe_binconv

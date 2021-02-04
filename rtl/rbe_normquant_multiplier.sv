/*
 * rbe_normquant_multiplier.sv
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
 *
 * TODO Description
 *
 */


module rbe_normquant_multiplier #(
  parameter int unsigned NMS  = rbe_package::NORM_MULT_SIZE,
  parameter int unsigned NAS  = rbe_package::NORM_ADD_SIZE,
  parameter int unsigned ACC  = rbe_package::ACCUMULATOR_SIZE,
  parameter int unsigned PIPE = 0
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,
  input  logic                      test_mode_i,
  input  logic                      clear_i,
  input  logic                      enable_i,
  input  logic signed [NMS:0]       norm_mult_signed_i,
  input  logic signed [NAS-1:0]     norm_add_i,
  input  logic signed [ACC-1:0]     accumulator_i,
  output logic signed [NMS+ACC-1:0] product_o
);

  logic [NMS+ACC-1:0] product_d, product_q;

  // compute sum-dot-product
  assign product_d = norm_mult_signed_i * accumulator_i + norm_add_i;

  // add pipeline stage if needed
  if(PIPE == 1) begin : pipe_gen
    always_ff@(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        product_q <= '0;
      end
      else if(clear_i) begin
        product_q <= '0;
      end
      else if(enable_i) begin
        product_q <= product_d;
      end
    end
    assign product_o = product_q;
  end
  else begin : no_pipe_gen
    assign product_o = product_d;
  end

endmodule // rbe_normquant_multiplier

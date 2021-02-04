/*
 * rbe_normquant.sv
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


module rbe_normquant #(
  parameter int unsigned NMS = rbe_package::NORM_MULT_SIZE,
  parameter int unsigned NAS = rbe_package::NORM_ADD_SIZE,
  parameter int unsigned ACC = rbe_package::ACCUMULATOR_SIZE,
  parameter int unsigned QNT = 32,
  parameter int unsigned PIPE = 1,
  parameter int unsigned OUTPUT_REGISTER = 0
) (
  // global signals
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   test_mode_i,
  // local clear
  input  logic                   clear_i,
  // normalization parameters
  input  logic unsigned [NMS-1:0] norm_mult_i,
  input  logic signed   [NAS-1:0] norm_add_i,
  // accumulation
  input  logic signed   [ACC-1:0] accumulator_i,
  output logic signed   [ACC-1:0] accumulator_o,
  // control channel
  input  rbe_package::ctrl_normquant_t  ctrl_i,
  output rbe_package::flags_normquant_t flags_o
);


  /////////////////////////
  // SIGNAL DECLARATIONS //
  /////////////////////////

  logic signed [NMS:0]       norm_mult_signed;
  logic signed [NMS+ACC-1:0] product;
  logic [NMS+ACC-1:0]        shifted;
  logic [ACC-1:0]            accumulator_d;
  logic [QNT-1:0]            accumulator_q;


  ///////////////////////////
  // MULTIPLY + ACCUMULATE //
  ///////////////////////////

  assign norm_mult_signed = {1'b0, norm_mult_i};

  rbe_normquant_multiplier #(
    .NMS  ( NMS  ),
    .NAS  ( NAS  ),
    .ACC  ( ACC  ),
    .PIPE ( PIPE )
  ) i_multiplier (
    .clk_i              ( clk_i            ),
    .rst_ni             ( rst_ni           ),
    .test_mode_i        ( test_mode_i      ),
    .clear_i            ( clear_i          ),
    .enable_i           ( 1'b1             ),
    .norm_mult_signed_i ( norm_mult_signed ),
    .norm_add_i         ( norm_add_i       ),
    .accumulator_i      ( accumulator_i    ),
    .product_o          ( product          )
  );

  assign shifted = product >>> ctrl_i.right_shift;


  /////////////////////////
  // SATURATION HANDLING //
  /////////////////////////

  logic [NMS+ACC-2:0] sat_big_or_shifted;
  logic [NMS+ACC-2:0] sat_big_nand_shifted;

  for (genvar ii=8; ii<NMS+ACC-1; ii++) begin
    assign sat_big_or_shifted[ii]   =  shifted[ii];
    assign sat_big_nand_shifted[ii] = ~shifted[ii];
  end

  for (genvar ii=0; ii<8; ii++) begin
    assign sat_big_or_shifted[ii]   = (ctrl_i.qa_out <= ii) ?  shifted[ii] : 1'b0;
    assign sat_big_nand_shifted[ii] = (ctrl_i.qa_out <= ii) ? ~shifted[ii] : 1'b0;
  end

  always_comb
  begin
    accumulator_d = shifted[ACC-1:0];
    if(ctrl_i.relu & shifted[NMS+ACC-1])
      accumulator_d = '0; // neg or sat-
    else if (~shifted[NMS+ACC-1] & (|(sat_big_or_shifted))) begin
      accumulator_d = '1; // sat+
    end
    else if (shifted[NMS+ACC-1] & (|(sat_big_nand_shifted))) begin
      accumulator_d = '0;
      accumulator_d[ctrl_i.qa_out - 1] = 1'b1; // sat-
    end
  end


  //////////////////////
  // OUTPUT REGISTERS //
  //////////////////////

  if(OUTPUT_REGISTER) begin : output_register_gen

    always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        accumulator_q <= '0;
        flags_o.ready <= 1'b0;
      end
      else if(clear_i) begin
        accumulator_q <= '0;
        flags_o.ready <= 1'b0;
      end
      else if(ctrl_i.start) begin
        accumulator_q <= accumulator_d[QNT-1:0];
        flags_o.ready <= 1'b1;
      end
    end

  end // output_register_gen
  else begin : no_output_register_gen

    assign accumulator_q = accumulator_d[QNT-1:0];
    assign flags_o.ready = 1'b1;

  end // no_output_register_gen


  ///////////////////////
  // OUTPUT ASSIGNMENT //
  ///////////////////////

  always_comb
  begin
    accumulator_o          = '0;
    accumulator_o[QNT-1:0] = accumulator_q;
  end

endmodule // rbe_normquant

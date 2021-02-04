/*
 * rbe_accumulators_scm.sv
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
 * This contains the standard-cell memories (latches) used to implement
 * the accumulators. It is derived from the register_file_2r_1w_all
 * SCM in the scm repository, released under SolderPad license.
 * The main difference is that these registers can be cleared.
 *
 */

module rbe_accumulators_scm
#(
  parameter int unsigned ADDR_WIDTH   = 5,
  parameter int unsigned DATA_WIDTH   = 32,
  parameter int unsigned NUM_WORDS    = 2**ADDR_WIDTH,
  parameter int unsigned WIDTH_FACTOR = 4
)
(
  input  logic                               clk_i,
  input  logic                               rst_ni,
  input  logic                               clear_i,
  input  logic                               test_mode_i,
  input  logic                               wide_enable_i,

  // Read port
  input  logic                               re_i,
  input  logic [ADDR_WIDTH-1:0]              raddr_i,
  output logic [DATA_WIDTH-1:0]              rdata_o,
  output logic [WIDTH_FACTOR*DATA_WIDTH-1:0] rdata_wide_o,

  // Write port
  input  logic                               we_i,
  input  logic                               we_all_i,
  input  logic [ADDR_WIDTH-1:0]              waddr_i,
  input  logic [DATA_WIDTH-1:0]              wdata_i,
  input  logic [WIDTH_FACTOR*DATA_WIDTH-1:0] wdata_wide_i,

  output logic [NUM_WORDS-1:0][DATA_WIDTH-1:0] accumulators_o
);

  /////////////
  // SIGNALS //
  /////////////

  // Read address register, located at the input of the address decoder
  logic [NUM_WORDS-1:0][DATA_WIDTH-1:0] accumulators;
  logic [NUM_WORDS-1:0]  waddr_onehot;
  logic [NUM_WORDS-1:0]  clk_we;

  logic [WIDTH_FACTOR*DATA_WIDTH-1:0]      rdata_q;
  logic [WIDTH_FACTOR-1:0][DATA_WIDTH-1:0] wdata_q;

  logic clk_gated;

  //////////////
  // CLK GATE //
  //////////////

  cluster_clock_gating i_cg_we_global (
    .clk_o     ( clk_gated      ),
    .en_i      ( we_i | clear_i ),
    .test_en_i ( test_mode_i    ),
    .clk_i     ( clk_i          )
  );

  ////////////////////
  // WDATA SAMPLING //
  ////////////////////

  for(genvar ii=0; ii<WIDTH_FACTOR; ii++) begin
    always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        wdata_q[ii] <= '0;
      else if(clear_i)
        wdata_q[ii] <= '0;
      else if((we_i | we_all_i) &  wide_enable_i)
        wdata_q[ii] <= wdata_wide_i[(ii+1)*DATA_WIDTH-1:ii*DATA_WIDTH];
      else if((we_i | we_all_i) & ~wide_enable_i)
        wdata_q[ii] <= wdata_i;
    end
  end

  ///////////////////
  // SCM (LATCHES) //
  ///////////////////

  // use the sampled address to select the correct rdata_o
  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni)
      rdata_q[DATA_WIDTH-1:0] <= '0;
    else if(clear_i)
      rdata_q[DATA_WIDTH-1:0] <= '0;
    else if(re_i) begin
      rdata_q[DATA_WIDTH-1:0] <= accumulators[raddr_i];
    end
  end


  for(genvar ii=1; ii<WIDTH_FACTOR; ii++) begin

    logic [ADDR_WIDTH-1:0] raddr_wide;
    assign raddr_wide = raddr_i + ii;

    always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        rdata_q[(ii+1)*DATA_WIDTH-1:ii*DATA_WIDTH] <= '0;
      else if(clear_i)
        rdata_q[(ii+1)*DATA_WIDTH-1:ii*DATA_WIDTH] <= '0;
      else if(re_i & wide_enable_i) begin
        rdata_q[(ii+1)*DATA_WIDTH-1:ii*DATA_WIDTH] <= accumulators[raddr_wide];
      end
    end

  end

  assign rdata_o      = rdata_q[DATA_WIDTH-1:0];
  assign rdata_wide_o = rdata_q;


  // decode
  for(genvar ii=0; ii<NUM_WORDS/WIDTH_FACTOR; ii++) begin : WADDR_DECODE_ITER_1
    for(genvar jj=0; jj<WIDTH_FACTOR; jj++) begin : WADDR_DECODE_ITER_0

      logic [ADDR_WIDTH-1:0] idx_ii_jj, idx_ii;
      assign idx_ii_jj = ii*WIDTH_FACTOR+jj;
      assign idx_ii    = ii*WIDTH_FACTOR;

      always_comb
      begin : waddr_decoding
        if((wide_enable_i==1'b1) && (we_i==1'b1) && (waddr_i == idx_ii))
          waddr_onehot[ii*WIDTH_FACTOR+jj] = 1'b1;
        else if((we_i==1'b1) && (waddr_i == idx_ii_jj))
          waddr_onehot[ii*WIDTH_FACTOR+jj] = 1'b1;
        else if(we_all_i==1'b1)
          waddr_onehot[ii*WIDTH_FACTOR+jj] = 1'b1;
        else
          waddr_onehot[ii*WIDTH_FACTOR+jj] = clear_i;
      end

    end // WADDR_DECODE_ITER_0
  end // WADDR_DECODE_ITER_1


  // generate one clock-gating cell for each register element
  for(genvar ii=0; ii<NUM_WORDS; ii++) begin : CG_CELL_WORD_ITER
    cluster_clock_gating i_cg (
      .clk_o     ( clk_we[ii]       ),
      .en_i      ( waddr_onehot[ii] ),
      .test_en_i ( test_mode_i      ),
      .clk_i     ( clk_i            )
    );
  end // CG_CELL_WORD_ITER


  for(genvar ii=0; ii<NUM_WORDS/WIDTH_FACTOR; ii++) begin : LATCH_ITER_1
    for(genvar jj=0; jj<WIDTH_FACTOR; jj++) begin : LATCH_ITER_0

      always_latch
      begin : latch_wdata
        if( clk_we[ii*WIDTH_FACTOR+jj] ) begin
          accumulators[ii*WIDTH_FACTOR+jj] = clear_i ? '0 : wdata_q[jj];
        end
      end

    end // LATCH_ITER_0
  end // LATCH_ITER_1

  assign accumulators_o = accumulators;

endmodule // rbe_accumulators_scm

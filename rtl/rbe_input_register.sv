/*
 * rbe_input_register.sv
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
 * The feature register is a latch-based buffer which is used in two states:
 *  - FR_LOAD: loading a BW-bit feature vector
 *  - FR_EXTRACT: for nof cycles the feature vector is available for computation
 * In the FR_LOAD phase, data is packed in 32 bit words, each representing a single
 * pixel across 32 channels / dimensions.
 * To implement an FxF filter, the entire set of input features per each window
 * position is loaded F*F times from TCDM. This is acceptable as its lifetime
 * inside the accelerator will be of nof cycles, where nof is typically >= nif.
 *
 */


module rbe_input_register #(
  parameter int unsigned TP       = rbe_package::BINCONV_TP,    // number of input elements processed per cycle
  parameter int unsigned CNT_SIZE = rbe_package::VLEN_CNT_SIZE, // counter size
  parameter int unsigned NPR      = 1,                          // number of registers
  parameter int unsigned LATCH    = 0
) (
  // global signals
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         test_mode_i,
  // local enable and clear
  input  logic                         enable_i,
  input  logic                         clear_i,
  // input / output streams
  hwpe_stream_intf_stream.sink         feat_i,
  hwpe_stream_intf_stream.source       feat_o,
  // control and flags
  input  rbe_package::ctrl_feat_buf_t  ctrl_feat_buf_i,
  output rbe_package::flags_feat_buf_t flags_feat_buf_o
);

  /////////////////////////
  // SIGNAL DECLARATIONS //
  /////////////////////////

  // Standard-cell memory based feature register
  logic [NPR-1:0][TP-1:0] input_buffer_rdata;
  logic [NPR-1:0]         input_buffer_we;
  logic [NPR-1:0]         input_buffer_clk;
  logic [NPR-1:0]         input_buffer_we_load;
  logic [NPR-1:0][TP-1:0] input_buffer_wdata;

  // Finite-state machine + counters
  rbe_package::state_feat_buf_t     fsm_cs, fsm_ns;

  logic                vlen_cnt_clr, vlen_cnt_gl_en, vlen_cnt_en;
  logic [CNT_SIZE-1:0] vlen_cnt;
  logic [CNT_SIZE-1:0] vlen_cnt_next;

  // Buffer bank write enable gating
  always_comb
  begin : input_buffer_we_mux
    input_buffer_we = '0;
    if(fsm_cs == rbe_package::FR_LOAD) begin
      input_buffer_we = input_buffer_we_load;
    end
  end

  for(genvar ii=0; ii<NPR; ii++) begin : input_buffer_gen

    // The write enable signal is converted to a clock
    // pulse on the negative phase of clk_i, that is then
    // used to sample input_buffer_wdata.
    cluster_clock_gating i_scm_feat_gate (
      .clk_i     ( clk_i                ),
      .test_en_i ( test_mode_i          ),
      .en_i      ( input_buffer_we [ii] ),
      .clk_o     ( input_buffer_clk[ii] )
    );

    always_ff @(posedge input_buffer_clk[ii] or negedge rst_ni)
    begin
      if(~rst_ni) begin
        input_buffer_rdata[ii] <= '0;
      end
      else begin
        input_buffer_rdata[ii] <= input_buffer_wdata[ii];
      end
    end

  end // input_buffer_gen


  /* FR_LOAD mode */
  always_comb
  begin : feat_buf_load_scatter
    input_buffer_wdata   = '0;
    input_buffer_we_load = '0;
    input_buffer_wdata   = feat_i.data;
    input_buffer_we_load = feat_i.valid;
  end

  /* FR_EXTRACT mode */
  assign feat_o.data = input_buffer_rdata[0];
  assign feat_o.strb = '1; // no strobes here

  /* control */
  // finite-state machine + buffer virtual length counter
  always_ff @(posedge clk_i or negedge rst_ni)
  begin : fsm_seq
    if(~rst_ni)
      fsm_cs <= rbe_package::FR_IDLE;
    else if(enable_i)
      fsm_cs <= fsm_ns;
  end

  // Main FSM
  always_comb
  begin : fsm_comb
    fsm_ns         = fsm_cs;
    feat_i.ready   = 1'b0;
    feat_o.valid   = 1'b0;
    vlen_cnt_clr   = 1'b1;
    vlen_cnt_gl_en = 1'b0;

    case (fsm_cs)
      // in FR_IDLE state, wait for a FR_LOAD / FR_EXTRACT command
      rbe_package::FR_IDLE: begin
        if(ctrl_feat_buf_i.goto_load)
          fsm_ns = rbe_package::FR_LOAD;
        else if(ctrl_feat_buf_i.goto_extract)
          fsm_ns = rbe_package::FR_EXTRACT;
      end

      // in FR_LOAD state, raise the ready for the stream hs until the buffer virtual length vlen has been reached
      rbe_package::FR_LOAD: begin
        feat_i.ready = 1'b1;
        vlen_cnt_gl_en = 1'b1;
        vlen_cnt_clr = 1'b0;
        if((feat_i.valid == 1'b1) && ({1'b0, vlen_cnt} == ctrl_feat_buf_i.i_vlen-1)) begin
          fsm_ns = rbe_package::FR_IDLE; // an intermediate FR_IDLE state before going to FR_EXTRACT is necessary
                                         // in any case due to the way the latch-based register works
          vlen_cnt_clr = 1'b1;
        end
      end

      // in FR_EXTRACT state, raise the valid for the feat hs until the buffer virtual length vlen has been reached
      rbe_package::FR_EXTRACT: begin
        feat_o.valid = 1'b1;
        vlen_cnt_gl_en = 1'b1;
        vlen_cnt_clr = 1'b0;
        if((feat_o.ready == 1'b1) && ({1'b0, vlen_cnt} == ctrl_feat_buf_i.o_vlen)) begin // should it be the same vlen? not really, but may be the same param
          fsm_ns = rbe_package::FR_IDLE;
          vlen_cnt_clr = 1'b1;
        end
      end

      default : begin
        if(ctrl_feat_buf_i.goto_load)
          fsm_ns = rbe_package::FR_LOAD;
        else if(ctrl_feat_buf_i.goto_extract)
          fsm_ns = rbe_package::FR_EXTRACT;
      end

    endcase
  end

  // virtual length counter (counts words of BP*32 size in FR_LOAD mode and, for now, also in FR_EXTRACT mode)
  assign vlen_cnt_en = (fsm_cs == rbe_package::FR_LOAD)    ? feat_i.valid & feat_i.ready :
                       (fsm_cs == rbe_package::FR_EXTRACT) ? feat_o.valid & feat_o.ready : '0;

  assign vlen_cnt_next = (vlen_cnt_clr == 1'b0) ? vlen_cnt + 1 : '0;

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : vlen_counter
    if(~rst_ni)
      vlen_cnt <= '0;
    else if (enable_i & vlen_cnt_gl_en) begin
      if (vlen_cnt_clr == 1'b1)
        vlen_cnt <= '0;
      else if(vlen_cnt_en)
        vlen_cnt <= vlen_cnt_next;
    end
  end

  // broadcast current state for external controller
  assign flags_feat_buf_o.state = fsm_cs;

endmodule // rbe_input_register

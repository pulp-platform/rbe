/*
 * rbe_accumulator_normquant.sv
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


module rbe_accumulator_normquant
  import rbe_package::*;
#(
  parameter  int unsigned TP             = rbe_package::BINCONV_TP,            // output filter size in bits/cycle
  parameter  int unsigned AP             = rbe_package::ACCUMULATOR_PARAMETER, // number of accumulators
  parameter  int unsigned ACC            = rbe_package::ACCUMULATOR_SIZE,
  parameter  int unsigned CNT            = rbe_package::VLEN_CNT_SIZE,
  parameter  int unsigned PIPE_NORMQUANT = 1,
  localparam int unsigned BW             = rbe_package::BANDWIDTH,
  localparam int unsigned N_MULT_SIZE    = rbe_package::NORM_MULT_SIZE,
  localparam int unsigned N_ADD_SIZE     = rbe_package::NORM_ADD_SIZE,
  localparam int unsigned NMAS           = N_MULT_SIZE+N_ADD_SIZE
) (
  // global signals
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   test_mode_i,
  // local enable & clear
  input  logic                   enable_i,
  input  logic                   clear_i,
  // incoming psums
  hwpe_stream_intf_stream.sink   conv_i,
  // incoming normalization parameters
  hwpe_stream_intf_stream.sink   norm_i,
  // output features + handshake
  hwpe_stream_intf_stream.source conv_o,
  // control and flags
  input  rbe_package::ctrl_aq_t  ctrl_i,
  output rbe_package::flags_aq_t flags_o
);

  /////////////////////////
  // SIGNAL DECLARATIONS //
  /////////////////////////

  logic signed [2**$clog2(AP)-1:0][ACC-1:0] all_accumulators;
  logic signed [ACC-1:0]                    accumulator_q;
  logic signed [4*ACC-1:0]                  accumulator_wide_q;
  logic signed [4*ACC-1:0]                  normalized_q;
  logic signed [ACC-1:0]                    offset_plus_d;
  logic signed [ACC-1:0]                    accumulator_plus_d;
  logic signed [ACC-1:0]                    offset_minus_d;
  logic                                     accumulator_we;
  logic                                     accumulator_re;
  logic                                     accumulator_clr;

  logic signed [ACC-1:0] offset_q, offset_d;
  logic offset_clr;

  logic signed [ACC-1:0] res_intermediate_q, res_intermediate_d;
  logic res_intermediate_clr, res_intermediate_sel;
  logic res_intermediate_sel_q, res_intermediate_sel_d;

  logic offset_en, offset_en_d, offset_en_q;
  logic offset_state, offset_state_d, offset_state_q;

  logic [$clog2(AP)-1:0] waddr;
  logic [$clog2(AP)-1:0] raddr;
  logic signed [ACC-1:0] wdata;

  logic we_all, we;

  // counter for accumulators
  logic [CNT-1:0] acc_cnt;
  logic [CNT-1:0] acc_cnt_q;
  logic [CNT-1:0] acc_cnt_next;
  logic           acc_cnt_clear, acc_cnt_clear_small;
  logic [CNT-1:0] full_accumulation_cnt;
  logic [    2:0] bit_cnt, bit_cnt_next;
  logic           bit_cnt_clr, bit_cnt_en;

  logic norm_hs_q, norm_hs_int;

  logic done_accumulation_q;
  logic ctrl_clear_q;

  logic [$clog2(AP)-1:0] waddr_cnt_q, waddr_cnt_d;
  logic waddr_cnt_en, waddr_cnt_clr;

  logic [$clog2(AP)-1:0] raddr_cnt_q, raddr_cnt_d;
  logic raddr_cnt_en, raddr_cnt_clr;

  logic conv_valid_hs_q;
  logic first_accum_q, first_accum_set, first_accum_clear;

  logic [1:0] hs_count_q;
  logic hs_count_en, hs_count_clr;

  // fsm state
  state_aq_t fsm_cs, fsm_ns;

  /////////////////////////////
  // OFFSET SPECIAL HANDLING //
  /////////////////////////////

  // address counter
  assign waddr_cnt_en = (acc_cnt == {'0,(ctrl_i.n_accum-1)}) & conv_valid_hs_q;
  assign raddr_cnt_en = (res_intermediate_sel & conv_i.valid & !(raddr_cnt_q==31)) | (first_accum_q & conv_i.valid);

  // address assignment
  assign waddr =                            (fsm_cs == AQ_NORMQUANT) ? acc_cnt_q    << 2 : waddr_cnt_q;
  assign raddr =                            (fsm_cs == AQ_NORMQUANT) ? acc_cnt_next << 2 :
                 (fsm_ns == AQ_ACCUM_DONE | fsm_ns == AQ_NORMQUANT ) ? '0 :
                                                        raddr_cnt_en ? raddr_cnt_q + 1 :
                                                                       raddr_cnt_q;

  // compute next value to be added
  assign accumulator_plus_d  =   accumulator_q      + $signed(conv_i.data) + offset_q;
  assign offset_plus_d       =   res_intermediate_q + $signed(conv_i.data);
  assign offset_minus_d      = - res_intermediate_q + $signed(conv_i.data);

  // accumulator we/re
  assign accumulator_we = (offset_en == 1'b1)      ? 1'b0         : conv_valid_hs_q;
  assign accumulator_re = (fsm_cs == AQ_NORMQUANT) ? norm_i.valid : conv_valid_hs_q;

  assign we_all     = (offset_en == 1'b1)        ? offset_state  : 1'b0;
  assign we         = (fsm_cs == AQ_NORMQUANT)   ? norm_hs_int   : accumulator_we;

  // accumulator write data
  assign wdata      = (fsm_cs == AQ_NORMQUANT || fsm_cs == AQ_NORMQUANT_DONE) ? normalized_q   :
                      (offset_en == 1'b1 & offset_state == 1'b1)              ? offset_minus_d :
                      ((hs_count_q==2'b10) & conv_valid_hs_q)                 ? accumulator_q  :
                                                                                res_intermediate_q;

  // compute offset and temporal results in this register
  assign res_intermediate_d   = (offset_en == 1'b1 & offset_state == 1'b1)    ? offset_minus_d :
                                (res_intermediate_sel | (hs_count_q==2'b10))  ? accumulator_plus_d :
                                                                                offset_plus_d;

  assign res_intermediate_clr = (fsm_cs == AQ_ACCUM_DONE || fsm_cs == AQ_NORMQUANT || fsm_cs == AQ_NORMQUANT_DONE);
  assign res_intermediate_sel = res_intermediate_sel_q & (fsm_cs == AQ_ACCUMULATE | fsm_cs == AQ_ACCUM_DONE) ? '1 :
                                                      (waddr_cnt_en & conv_valid_hs_q & (fsm_cs == AQ_ACCUMULATE));

  // keep offset_en, offset_state and intermediate result selector if incoming data is not valid
  assign res_intermediate_sel_d = res_intermediate_sel & !conv_i.valid;
  assign offset_en_d            = ctrl_i.offset_en     & !conv_i.valid;
  assign offset_state_d         = ctrl_i.offset_state  & !conv_i.valid;

  assign offset_en    = (offset_en_q | (offset_state & !conv_i.valid)) ? '1 : ctrl_i.offset_en;
  assign offset_state = offset_state_q ? '1 : ctrl_i.offset_state;

  // store current offset value in offset_q
  assign offset_d = we_all ? wdata : '0;

  //first_accum_set
  assign first_accum_set   = we_all & conv_i.valid;
  assign first_accum_clear = (first_accum_q & conv_i.valid) ? 1'b1 : 1'b0;

  // count first two handshakes
  assign hs_count_en  = conv_i.valid & conv_i.ready & !(hs_count_q==2'b11);
  assign hs_count_clr = (fsm_cs==AQ_ACCUM_DONE) | (fsm_cs==AQ_NORMQUANT_DONE) | (fsm_cs==AQ_NORMQUANT);

  ////////////////////////////
  // ACCUMULATOR SCM MODULE //
  ////////////////////////////

  // one additional entry for the offset computation
  // (where written in 1 cycle in the next read, subtracted and written to all addresses
  logic wide_enable;
  assign wide_enable = (fsm_cs == AQ_NORMQUANT) || (fsm_cs == AQ_ACCUM_DONE) ? 1'b1 : 1'b0;

  rbe_accumulators_scm #(
    .ADDR_WIDTH ( $clog2(AP) ),
    .DATA_WIDTH ( ACC        )
  ) i_accumulators (
    .clk_i             ( clk_i                     ),
    .rst_ni            ( rst_ni                    ),
    .clear_i           ( accumulator_clr | clear_i ),
    .test_mode_i       ( test_mode_i               ),
    .wide_enable_i     ( wide_enable               ),
    .re_i              ( enable_i                  ),
    .raddr_i           ( raddr                     ),
    .rdata_o           ( accumulator_q             ),
    .rdata_wide_o      ( accumulator_wide_q        ),
    .we_all_i          ( '0                        ),
    .we_i              ( we                        ),
    .waddr_i           ( waddr                     ),
    .wdata_i           ( wdata                     ),
    .wdata_wide_i      ( normalized_q              ),
    .accumulators_o    ( all_accumulators          )
  );

  logic [BW-1:0] norm_data;
  logic [3:0][N_MULT_SIZE-1:0] norm_mult;
  logic [3:0][ N_ADD_SIZE-1:0] norm_add;
  ctrl_normquant_t ctrl_normquant;
  flags_normquant_t flags_normquant[4];

  // four norm params per bigword, in the following arrangement
  // +-------------------------------------------------------------------------------------------------------+
  // | ADD3           | MUL3   | ADD2           | MUL2   | ADD1           | MUL1   | ADD0           | MUL0   |
  // +-------------------------------------------------------------------------------------------------------+
  // | 191:160        | 159:144| 143:112        | 111:96 | 95:64          | 63:48  | 47:16          | 15:0   |
  // +-------------------------------------------------------------------------------------------------------+
  assign norm_data = norm_i.data;

  for(genvar ii=0; ii<4; ii++) begin : norm_mult_add_gen

    assign norm_mult[ii] = norm_data[ii*NMAS+N_MULT_SIZE-1:ii*NMAS];
    assign norm_add [ii] = norm_data[ii*NMAS+NMAS-1:ii*NMAS+N_MULT_SIZE];

    rbe_normquant #(
      .ACC  ( ACC            ),
      .PIPE ( PIPE_NORMQUANT )
    ) i_normquant (
      .clk_i         ( clk_i                                  ),
      .rst_ni        ( rst_ni                                 ),
      .test_mode_i   ( test_mode_i                            ),
      .clear_i       ( clear_i                                ),
      .norm_mult_i   ( norm_mult          [ii]                ),
      .norm_add_i    ( norm_add           [ii]                ),
      .accumulator_i ( accumulator_wide_q [(ii+1)*32-1:ii*32] ),
      .accumulator_o ( normalized_q       [(ii+1)*32-1:ii*32] ),
      .ctrl_i        ( ctrl_normquant                         ),
      .flags_o       ( flags_normquant    [ii]                )
    );

  end // norm_mult_add_gen


  always_comb
  begin
    ctrl_normquant = ctrl_i.ctrl_normquant;
    ctrl_normquant.start = fsm_cs == AQ_NORMQUANT ? 1'b1 : 1'b0;
  end

  // Stream-out construction
  logic [31:0] conv_data;

  logic [$clog2(QUANT_CNT_SIZE):0] qa_out_m_1;
  logic [$clog2(QUANT_CNT_SIZE):0] stream_limit;

  assign qa_out_m_1 = {'0, (ctrl_i.qa_out-1)};
  assign stream_limit = qa_out_m_1[3:2]=='0 ? 4 : 8;

  always_comb
  begin
    conv_data = '0;
    for(int i=0; i<TP; i++) begin
      if(bit_cnt < ctrl_i.qa_out) begin
        conv_data[i] = all_accumulators[i][bit_cnt];
      end
      else begin
        conv_data[i] = '0;
      end
    end
  end

  assign conv_o.data = conv_data;

  // Ready propagation
  assign conv_i.ready = (fsm_cs == AQ_ACCUMULATE) || (fsm_cs == AQ_IDLE) ? 1'b1 : 1'b0;
  assign norm_i.ready = (fsm_cs == AQ_NORMQUANT) ? norm_i.valid : 1'b0;


  //////////////
  // COUNTERS //
  //////////////

  // read and write address in Registers
  assign acc_cnt_next = (clear_i | (acc_cnt_clear_small & accumulator_re) | acc_cnt_clear) ? '0 :
                                                                            accumulator_re ? acc_cnt + 1 :
                                                                                             acc_cnt;

  // accumulator address counter
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : accumulator_counter
      if(~rst_ni) begin
        acc_cnt <= '0;
      end
      else if(enable_i) begin
        if(clear_i)
          acc_cnt <= '0;
        else
          acc_cnt <= acc_cnt_next;
      end
    end

  // if normquant is pipelined, delay acc_cnt_q
  if(PIPE_NORMQUANT) begin : pipe_normquant
    always_ff @(posedge clk_i or negedge rst_ni)
      begin : accumulator_counter_q
        if(~rst_ni) begin
          acc_cnt_q <= '0;
        end
        else if(enable_i) begin
          if(clear_i | acc_cnt_clear)
            acc_cnt_q <= '0;
          else if(norm_i.ready)
            acc_cnt_q <= acc_cnt;
        end
      end
    assign norm_hs_int = norm_hs_q;
  end
  else begin : no_pipe_normquant
    assign acc_cnt_q = acc_cnt;
    assign norm_hs_int = norm_i.valid;
  end

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        res_intermediate_sel_q   <= '0;
        offset_en_q              <= '0;
        offset_state_q           <= '0;
      end
      else if(enable_i) begin
        if(clear_i) begin
          res_intermediate_sel_q <= '0;
          offset_en_q            <= '0;
          offset_state_q         <= '0;
        end
        else begin
          res_intermediate_sel_q <= res_intermediate_sel_d;
          offset_en_q            <= offset_en_d;
          offset_state_q         <= offset_state_d;
        end
      end
    end

  // valid handshake counter
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : hs_counter
      if(~rst_ni) begin
        hs_count_q <= '0;
      end
      else if(hs_count_clr | clear_i)
        hs_count_q <= '0;
      else if(hs_count_en)
        hs_count_q <= hs_count_q + 1;
    end

  // waddr counter
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : waddr_counter
      if(~rst_ni) begin
        waddr_cnt_q <= '0;
      end
      else if(waddr_cnt_clr | clear_i)
        waddr_cnt_q <= '0;
      else if(waddr_cnt_en)
        waddr_cnt_q <= waddr_cnt_q + 1;
    end

  // raddr counter
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : raddr_counter
      if(~rst_ni) begin
        raddr_cnt_q <= '0;
      end
      else if(raddr_cnt_clr | clear_i)
        raddr_cnt_q <= '0;
      else if(raddr_cnt_en)
        raddr_cnt_q <= raddr_cnt_q + 1;
    end

  // bit counter
  assign bit_cnt_next = bit_cnt + 1;

  always_ff @(posedge clk_i or negedge rst_ni)
    begin : bit_counter
      if(~rst_ni) begin
        bit_cnt <= '0;
      end
      else if(enable_i) begin
        if(clear_i | bit_cnt_clr)
          bit_cnt <= '0;
        else if(bit_cnt_en)
          bit_cnt <= bit_cnt_next;
      end
    end

  logic full_accumulation_cnt_en;
  assign full_accumulation_cnt_en = (ctrl_i.offset_en == 1'b1) ? 1'b0 : conv_valid_hs_q;

  // counter of number of accumulation done
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : vlen_counter
      if(~rst_ni) begin
        full_accumulation_cnt <= '0;
      end
      else if(enable_i) begin
        if(clear_i)
          full_accumulation_cnt <= '0;
        else if((fsm_cs == AQ_ACCUM_DONE) || (fsm_cs == AQ_NORMQUANT))
          full_accumulation_cnt <= '0;
        else if(full_accumulation_cnt_en & (fsm_cs==AQ_ACCUMULATE))
          full_accumulation_cnt <= full_accumulation_cnt + 1;
      end
    end


  ///////////////////
  // FSM REGISTERS //
  ///////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
    begin : fsm_seq
      if(~rst_ni) begin
        fsm_cs <= AQ_IDLE;
      end
      else if(enable_i) begin
        if(clear_i)
          fsm_cs <= AQ_IDLE;
        else
          fsm_cs <= fsm_ns;
      end
    end

    assign acc_cnt_clear = !((fsm_cs == AQ_ACCUMULATE) | (fsm_cs == AQ_NORMQUANT)) ? 1'b1 : 1'b0;
    assign acc_cnt_clear_small = !((fsm_cs == AQ_ACCUMULATE) | (fsm_cs == AQ_NORMQUANT)) ? 1'b1 :
                        ((acc_cnt == {'0,ctrl_i.n_accum-1}) & !(fsm_cs == AQ_NORMQUANT)) ? 1'b1 :
                                                                                           1'b0;
  //////////////
  // FSM Code //
  //////////////

  always_comb
    begin : fsm_comb
      fsm_ns = fsm_cs;
      bit_cnt_clr   = '0;
      bit_cnt_en    = '0;
      waddr_cnt_clr = '0;
      raddr_cnt_clr = '0;

      case(fsm_cs)

        AQ_IDLE : begin
          waddr_cnt_clr = 1'b1;
          if(accumulator_we)
            fsm_ns = AQ_ACCUMULATE;
        end

        AQ_ACCUMULATE : begin
          if((accumulator_we == 1'b1) && (full_accumulation_cnt == ctrl_i.full_accumulation_len-1))
            fsm_ns = AQ_ACCUM_DONE;
        end

        AQ_ACCUM_DONE : begin
          if(ctrl_i.goto_norm) begin
            fsm_ns = AQ_NORMQUANT;
            waddr_cnt_clr = 1'b1;
            raddr_cnt_clr = 1'b1;
          end
          else if(ctrl_i.goto_accum) begin
            waddr_cnt_clr = 1'b1;
            raddr_cnt_clr = 1'b1;
            if(accumulator_we) begin
              fsm_ns = AQ_ACCUMULATE;
            end
            else begin
              fsm_ns = AQ_IDLE;
            end
          end
        end

        AQ_NORMQUANT : begin
          waddr_cnt_clr = 1'b1;
          raddr_cnt_clr = 1'b1;
          if(acc_cnt_q == TP/4-1) begin
            fsm_ns = AQ_NORMQUANT_DONE;
          end
        end

        AQ_NORMQUANT_DONE : begin
          fsm_ns = AQ_STREAMOUT;
          waddr_cnt_clr = 1'b1;
          raddr_cnt_clr = 1'b1;
        end

        AQ_STREAMOUT : begin
          waddr_cnt_clr = 1'b1;
          raddr_cnt_clr = 1'b1;
          bit_cnt_en = conv_o.ready;
          if((bit_cnt == {'0,(stream_limit-1)}) & bit_cnt_en) begin
            fsm_ns = AQ_STREAMOUT_DONE;
            bit_cnt_clr = 1'b1;
          end
        end

        AQ_STREAMOUT_DONE : begin
          fsm_ns = AQ_IDLE;
        end

      endcase
    end
  assign conv_o.strb = '1; // FIXME

  ///////////////////////
  // GENERAL REGISTERS //
  ///////////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        ctrl_clear_q <= '0;
      end
      else if(clear_i) begin
        ctrl_clear_q <= '0;
      end
      else begin
        ctrl_clear_q <= ctrl_i.clear;
      end
    end

  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      done_accumulation_q <= '0;
    end
    else if(clear_i) begin
      done_accumulation_q <= '0;
    end
    else if(enable_i) begin
      done_accumulation_q <= bit_cnt_clr;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      res_intermediate_q <= '0;
    end
    else if(clear_i | res_intermediate_clr) begin
      res_intermediate_q <= '0;
    end
    else if(enable_i & conv_i.valid) begin
      res_intermediate_q <= res_intermediate_d;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      offset_q <= '0;
    end
    else if(clear_i | offset_clr) begin
      offset_q <= '0;
    end
    else if(we_all) begin
      offset_q <= offset_d;
    end
  end

  // valid conv handshake register
  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        conv_valid_hs_q <= '0;
        norm_hs_q       <= '0;
      end
      else if(clear_i) begin
        conv_valid_hs_q <= '0;
        norm_hs_q       <= '0;
      end
      else begin
        conv_valid_hs_q <= (conv_i.valid & conv_i.ready);
        norm_hs_q       <= norm_i.valid & norm_i.ready;
      end
    end

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni) begin
        first_accum_q <= '0;
      end
      else if(clear_i | first_accum_clear) begin
        first_accum_q <= '0;
      end
      else if(first_accum_set) begin
        first_accum_q <= 1'b1;
      end
    end


  ////////////////////////
  // OUTPUT ASSIGNMENTS //
  ////////////////////////

  assign conv_o.valid     = (fsm_cs == AQ_STREAMOUT) ? 1'b1 : 1'b0;
  assign accumulator_clr  = clear_i | ctrl_clear_q | done_accumulation_q;
  assign flags_o.state    = fsm_cs;
  assign flags_o.norm_cnt = acc_cnt[1:0];
  assign flags_o.bit_cnt  = bit_cnt;

endmodule

/*
 * rbe_ctrl_fsm.sv
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


module rbe_ctrl_fsm
  import rbe_package::*;
#(
  parameter int unsigned TP = rbe_package::BINCONV_TP
) (
  // global signals
  input  logic                               clk_i,
  input  logic                               rst_ni,
  input  logic                               test_mode_i,
  input  logic                               clear_i,
  // ctrl & flags
  output ctrl_streamer_t                     ctrl_streamer_o,
  input  flags_streamer_t                    flags_streamer_i,
  output ctrl_engine_t                       ctrl_engine_o,
  input  flags_engine_t                      flags_engine_i,
  output hwpe_ctrl_package::ctrl_uloop_t     ctrl_uloop_o,
  input  hwpe_ctrl_package::flags_uloop_t    flags_uloop_i,
  output ctrl_ctrlmult_t                     ctrl_ctrlmult_o,
  input  flags_ctrlmult_t                    flags_ctrlmult_i,
  output hwpe_ctrl_package::ctrl_slave_t     ctrl_slave_o,
  input  hwpe_ctrl_package::flags_slave_t    flags_slave_i,
  input  hwpe_ctrl_package::ctrl_regfile_t   reg_file_i,
  input  ctrl_ctrlfsm_t                      ctrl_i
);

  /////////////
  // SIGNALS //
  /////////////

  // fsm signal
  state_rbectrl_e curr_rbectrl_state, next_rbectrl_state;

  // counter signals
  logic weight_sel_count_enable;
  logic weight_sel_count_clear;

  // offset signals
  logic          offset_state_q, offset_state_d;
  logic          offset_state_delayed_0_q, offset_state_delayed_1_q, offset_state_delayed_2_q;
  logic [32*8:0] ws_count_q, ws_count_d; // TODO parametrize
  logic          offset_en_delayed_0_q, offset_en_delayed_1_q, offset_en_delayed_2_q;

  // flag combination signals
  logic                                                    feat_buff_all_sr_idle;
  logic [   $clog2(NR_ACTIVATIONS):0][ NR_ACTIVATIONS-1:0] feat_buff_all_sr_idle_tree;

  logic                                                    all_aq_popcnt_done;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_popcnt_done_tree;

  logic                                                    all_aq_accumulate;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_accumulate_tree;

  logic                                                    all_aq_idle;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_idle_tree;

  logic                                                    all_aq_normquant;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_normquant_tree;

  logic                                                    all_aq_normquant_done;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_normquant_done_tree;

  logic                                                    all_aq_streamout;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_streamout_tree;

  logic                                                    all_aq_streamout_done;
  logic [$clog2(BINCONV_NR_COLUMN):0][BINCONV_NR_COLUMN:0] all_aq_streamout_done_tree;

  logic last_fr_idle;

  logic empty_cycles_cnt_q;
  logic empty_cycles_cnt_en, empty_cycles_cnt_clear;
  logic [$clog2(QUANT_CNT_SIZE):0] nr_empty_cycles_cnt_q, nr_empty_cycles_cnt_d;

  logic conv_cnt_q;
  logic conv_cnt_en, conv_cnt_clear;

  logic [1:0] norm_cnt_q;
  logic norm_cnt_clear, norm_cnt_en;

  logic weight_cnt_q;
  logic weight_cnt_clear, weight_cnt_en;

  logic [$clog2(RBE_ULOOP_NB_LOOPS)-1:0] flags_uloop_last_loop;
  logic                                  flags_uloop_valid_loop, flags_uloop_done_loop;
  logic                                  flags_uloop_last_clear, flags_uloop_last_en;

  logic [RBE_ULOOP_CNT_WIDTH+2-1:0] n_tiles_feat_cnt;
  logic n_tiles_feat_cnt_en, n_tiles_feat_cnt_clear;

  logic [RBE_ULOOP_CNT_WIDTH+2-1:0] n_tiles_conv_cnt;
  logic n_tiles_conv_cnt_en, n_tiles_conv_cnt_clear;

  //////////////////////////////
  // INPUT FLAGS COMBINATIONS //
  //////////////////////////////

  //////////////////////////////
  // aq_popcnt_done
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_popcnt_done_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_ACCUM_DONE);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_popcnt_done_tree[j+1][i/2] = all_aq_popcnt_done_tree[j][i] & all_aq_popcnt_done_tree[j][i+1];
    end
  end

  assign all_aq_popcnt_done = all_aq_popcnt_done_tree[($clog2(BINCONV_NR_COLUMN)-1)][0];

  //////////////////////////////
  // aq_accumulate
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_accumulate_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_ACCUMULATE);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_accumulate_tree[j+1][i/2] = all_aq_accumulate_tree[j][i] & all_aq_accumulate_tree[j][i+1];
    end
  end

  assign all_aq_accumulate = all_aq_accumulate_tree[$clog2(BINCONV_NR_COLUMN)-1][0];

  //////////////////////////////
  // aq_idle
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_idle_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_IDLE);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_idle_tree[j+1][i/2] = all_aq_idle_tree[j][i] & all_aq_idle_tree[j][i+1];
    end
  end

  assign all_aq_idle = all_aq_idle_tree[($clog2(BINCONV_NR_COLUMN)-1)][0];

  //////////////////////////////
  // aq_normquant
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_normquant_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_NORMQUANT);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_normquant_tree[j+1][i/2] = all_aq_normquant_tree[j][i] & all_aq_normquant_tree[j][i+1];
    end
  end

  assign all_aq_normquant = all_aq_normquant_tree[$clog2(BINCONV_NR_COLUMN)-1][0];

  //////////////////////////////
  // AQ_NORMQUANT_DONE
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_normquant_done_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_NORMQUANT_DONE);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_normquant_done_tree[j+1][i/2] = all_aq_normquant_done_tree[j][i] & all_aq_normquant_done_tree[j][i+1];
    end
  end

  assign all_aq_normquant_done = all_aq_normquant_done_tree[$clog2(BINCONV_NR_COLUMN)-1][0];

  //////////////////////////////
  // AQ_STREAMOUT
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_streamout_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_STREAMOUT);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_streamout_tree[j+1][i/2] = all_aq_streamout_tree[j][i] & all_aq_streamout_tree[j][i+1];
    end
  end

  assign all_aq_streamout = all_aq_streamout_tree[$clog2(BINCONV_NR_COLUMN)-1][0];

  //////////////////////////////
  // AQ_STREAMOUT_DONE
  for(genvar i=0; i<BINCONV_NR_COLUMN; i++) begin
    assign all_aq_streamout_done_tree[0][i] = (flags_engine_i.flags_engine_core.flags_aq[i].state == AQ_STREAMOUT_DONE);
  end

  for(genvar j=0; j<$clog2(BINCONV_NR_COLUMN); j++) begin
    for(genvar i=0; i<BINCONV_NR_COLUMN; i+=2) begin
      assign all_aq_streamout_done_tree[j+1][i/2] = all_aq_streamout_done_tree[j][i] & all_aq_streamout_done_tree[j][i+1];
    end
  end

  assign all_aq_streamout_done = all_aq_streamout_done_tree[$clog2(BINCONV_NR_COLUMN)-1][0];

  //////////////////////////////
  // FR_IDLE
  assign last_fr_idle = (flags_engine_i.flags_engine_core.flags_activation_reg[NR_ACTIVATIONS-1].state == FR_IDLE);


  //////////////////
  // FSM REGISTER //
  //////////////////
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : main_fsm_seq
      if(~rst_ni) begin
        curr_rbectrl_state  <= RBECTRL_IDLE;
      end
      else if(clear_i) begin
        curr_rbectrl_state  <= RBECTRL_IDLE;
      end
      else begin
        curr_rbectrl_state  <= next_rbectrl_state;
      end
    end

  ///////////////
  // FSM LOGIC //
  ///////////////
  always_comb
    begin : main_fsm_comb

      //////////////////////
      // DEFAULT MAPPINGS //
      //////////////////////

      // ========================================================================
      // Streamer
      // direct mappings - these have to be here due to blocking/non-blocking assignment
      // combination with the same ctrl_engine_o/ctrl_streamer_o variable
      // shift-by-3 due to conversion from bits to bytes
      // x / feat stream
      ctrl_streamer_o.feat_source_ctrl.addressgen_ctrl.word_length  = (ctrl_i.fs == 1) ? {'0,NR_A_STREAMS_MIN} : {'0,NR_A_STREAMS_MAX};
      ctrl_streamer_o.feat_source_ctrl.addressgen_ctrl.word_stride  = ctrl_i.iw*4;
      ctrl_streamer_o.feat_source_ctrl.addressgen_ctrl.line_stride  = ctrl_i.static_w_tiles_K_in_qa << 4;
      ctrl_streamer_o.feat_source_ctrl.addressgen_ctrl.line_length  = {'0,BINCONV_BLOCK_SIZE};
      ctrl_streamer_o.feat_source_ctrl.addressgen_ctrl.base_addr    = reg_file_i.hwpe_params[RBE_REG_X_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_X_OFFS] >> 3);

      if(ctrl_i.fs == 1) begin
        // W / weight stream
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.word_length  = (TP + 2);
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.word_stride  = (ctrl_i.qw * 4);
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.line_stride  = '0;
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.line_length  = (ctrl_i.qw * 4)*(TP + 2);
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.base_addr    = reg_file_i.hwpe_params[RBE_REG_W_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_W_OFFS]);
      end
      else begin //if(ctrl_i.fs==3) begin
        // W / weight stream
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.word_length  = (ctrl_i.qw * TP + 2);
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.word_stride  = 36;
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.line_stride  = '0;
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.line_length  = 36 * (ctrl_i.qw * TP + 2);
        ctrl_streamer_o.weight_source_ctrl.addressgen_ctrl.base_addr    = reg_file_i.hwpe_params[RBE_REG_W_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_W_OFFS]);
      end

      // norm stream
      ctrl_streamer_o.norm_source_ctrl.addressgen_ctrl.word_length  = 8;
      ctrl_streamer_o.norm_source_ctrl.addressgen_ctrl.word_stride  = 24;
      ctrl_streamer_o.norm_source_ctrl.addressgen_ctrl.line_stride  = '0;
      ctrl_streamer_o.norm_source_ctrl.addressgen_ctrl.line_length  = 8 * 24;
      ctrl_streamer_o.norm_source_ctrl.addressgen_ctrl.base_addr    = reg_file_i.hwpe_params[RBE_REG_NQ_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_NQ_OFFS] >> 3);

      // y / conv stream
      ctrl_streamer_o.conv_sink_ctrl.addressgen_ctrl.word_length  = {'0,NR_A_STREAMS_MIN};
      ctrl_streamer_o.conv_sink_ctrl.addressgen_ctrl.word_stride  = ctrl_i.ow*4;
      ctrl_streamer_o.conv_sink_ctrl.addressgen_ctrl.line_stride  = ctrl_i.static_w_tiles_K_out_qa_out << 4;
      ctrl_streamer_o.conv_sink_ctrl.addressgen_ctrl.line_length  = {'0,BINCONV_BLOCK_SIZE};
      ctrl_streamer_o.conv_sink_ctrl.addressgen_ctrl.base_addr    = (n_tiles_conv_cnt=={'0, 1'b1}) ? reg_file_i.hwpe_params[RBE_REG_Y_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_Y_OFFS] >> 3) + {'0, {'0, ctrl_i.ow<<4} } : reg_file_i.hwpe_params[RBE_REG_Y_ADDR] + (flags_uloop_i.offs[RBE_ULOOP_Y_OFFS] >> 3);

      // ========================================================================
      // engine
      ctrl_engine_o.fs = ctrl_i.fs;

      ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.ctrl_sop.operation_sel = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.ctrl_sop.inactive_mask = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.ctrl_sop.clear         = '0;

      ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.qa_tile_sel = !(ctrl_i.n_tiles_qa == {'0,1'b1}) ? {'0,n_tiles_feat_cnt[0]} : '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.qw = ctrl_i.qw;

      ctrl_engine_o.ctrl_engine_core.ctrl_array.fs = ctrl_i.fs;

      ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load     = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_extract  = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen        =  1;

      // ========================================================================
      // accumulator
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.clear      = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.qa_out     = ctrl_i.qa_out;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_norm  = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_accum = '0;

      ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.qa_out      = ctrl_i.qa_out;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.start       = '0;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.relu        = ctrl_i.relu;
      ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.right_shift = ctrl_i.shift;

      ctrl_engine_o.qa_out = ctrl_i.qa_out;

      if(ctrl_i.fs == 3) begin
        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.row_onehot_en = 9'b111111111;
        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.fs            = 3;

        ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.o_vlen = TP*ctrl_i.qw+1;

        ctrl_engine_o.ctrl_engine_core.ctrl_aq.offset_en               = (curr_rbectrl_state==RBECTRL_COMPUTE) ? offset_en_delayed_2_q    : '0;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.offset_state            = (curr_rbectrl_state==RBECTRL_COMPUTE) ? offset_state_delayed_2_q : '0;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.mask                    = '0;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.full_accumulation_len   = TP*ctrl_i.qw;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.single_accumulation_len = TP;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.n_accum                 = ctrl_i.qw;

        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.offset_state  = (curr_rbectrl_state==RBECTRL_COMPUTE) ? offset_state_q : '0;
      end
      else begin // ctrl_i.fs==1
        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.row_onehot_en = (9'b111111111 >> (9-ctrl_i.qw));
        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.fs            = 1;

        ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.o_vlen = TP+1;

        ctrl_engine_o.ctrl_engine_core.ctrl_aq.offset_en               = offset_en_delayed_2_q;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.offset_state            = offset_state_delayed_2_q;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.mask                    = '0;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.full_accumulation_len   = TP;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.single_accumulation_len = TP;
        ctrl_engine_o.ctrl_engine_core.ctrl_aq.n_accum                 = 1;

        ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.offset_state  = offset_state_q;
      end

      ws_count_d = ws_count_q + 1;

      // ========================================================================
      // slave
      ctrl_slave_o.done       = '0;
      ctrl_slave_o.evt        = '0;

      // ========================================================================
      // streamer
      ctrl_streamer_o.ld_which_mux_sel             = LD_FEAT_SEL;
      ctrl_streamer_o.ld_st_mux_sel                = '0;
      ctrl_streamer_o.feat_source_ctrl.req_start   = '0;
      ctrl_streamer_o.weight_source_ctrl.req_start = '0;
      ctrl_streamer_o.conv_sink_ctrl.req_start     = '0;
      ctrl_streamer_o.norm_source_ctrl.req_start   = '0;

      // ========================================================================
      // uloop
      ctrl_uloop_o.enable = '0;
      ctrl_uloop_o.clear  = '0;

      flags_uloop_last_en    = '0;
      flags_uloop_last_clear = '0;

      nr_empty_cycles_cnt_d = nr_empty_cycles_cnt_q;

      empty_cycles_cnt_en    = 1'b0;
      empty_cycles_cnt_clear = 1'b0;
      conv_cnt_en            = 1'b0;
      conv_cnt_clear         = 1'b0;
      norm_cnt_en            = 1'b0;
      norm_cnt_clear         = 1'b0;
      weight_cnt_en          = 1'b0;
      weight_cnt_clear       = 1'b0;

      n_tiles_feat_cnt_en    = 1'b0;
      n_tiles_feat_cnt_clear = 1'b0;

      n_tiles_conv_cnt_en    = 1'b0;
      n_tiles_conv_cnt_clear = 1'b0;

      // ========================================================================
      // general ctrl
      ctrl_ctrlmult_o.start    = '0;

      // ========================================================================
      // real finite-state machine
      next_rbectrl_state   = curr_rbectrl_state;

      // ========================================================================
      // helper signals
      weight_sel_count_enable = 0;
      weight_sel_count_clear  = 0;

      //////////////
      // FSM CODE //
      //////////////
      case(curr_rbectrl_state)

        // ========================================================================
        // IDLE before start
        RBECTRL_IDLE:
          begin
            // wait for a start signal
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
            ctrl_uloop_o.clear               = '1;          // reset/clear uloop

            if(flags_slave_i.start) begin
              ctrl_ctrlmult_o.start = 1'b1;
              next_rbectrl_state    = RBECTRL_WAITMULT;
            end
          end

        // ========================================================================
        // wait for multiplicaiton
        RBECTRL_WAITMULT:
          begin
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load

            // if mulitplication donw
            if(flags_ctrlmult_i.valid) begin
              next_rbectrl_state = RBECTRL_START;

              // clear all accumulation and sop registers
              for(int i=0; i<BINCONV_NR_COLUMN; i++) begin
                ctrl_engine_o.ctrl_engine_core.ctrl_aq.clear                                    = 1'b1;
                ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.ctrl_sop.clear = 1'b1;
              end
            end
          end

        // ========================================================================
        // update indices and load feature
        RBECTRL_START:
          begin

            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load

            // if streams are ready, send data request and set feat_registers into load mode
            if(flags_streamer_i.feat_source_flags.ready_start) begin

              next_rbectrl_state = RBECTRL_LOAD_FEAT;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen = 1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load = 1'b1;

              // request data
              ctrl_streamer_o.feat_source_ctrl.req_start   = 1'b1;
            end
            else begin
              next_rbectrl_state = RBECTRL_WAIT;
            end
          end

        // ========================================================================
        // load feature
        RBECTRL_LOAD_FEAT:
          begin
            // load the first feature, then go to the compute phase
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
            weight_cnt_clear                 = 1'b1;

            // for close to infinity stalls
            if (flags_streamer_i.feat_source_flags.ready_start) begin
              if( (flags_uloop_last_loop == 3'b001) | (flags_uloop_last_loop == 3'b000) ) begin
                n_tiles_feat_cnt_en = 1'b1;
              end

              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen    = 1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load = 1'b1;

              // request data
              ctrl_streamer_o.feat_source_ctrl.req_start = 1'b1;
            end

            // prepare accumulators
            ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_accum = 1'b1;

            if (flags_streamer_i.feat_source_flags.done) begin
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_extract = 1'b1;
              weight_cnt_clear                                                = 1'b1;
              next_rbectrl_state                                              = RBECTRL_COMPUTE;
            end
          end

        // ========================================================================
        // stream in data and compute results
        RBECTRL_COMPUTE:
          begin

            ctrl_streamer_o.ld_which_mux_sel = LD_WEIGHT_SEL; // select weight
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;          // select load

            if( (flags_streamer_i.weight_source_flags.done | flags_streamer_i.weight_source_flags.ready_start ) & (weight_cnt_q=={'0,1'b0})) begin
              ctrl_streamer_o.weight_source_ctrl.req_start                    = 1'b1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_extract = 1'b1;
              weight_cnt_en                                                   = 1'b1;
            end

            if( flags_streamer_i.weight_source_flags.done ) begin

              if( (n_tiles_feat_cnt==(ctrl_i.n_tiles_K_in_qa-1)) ) begin
                ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_norm            = 1'b1;
                ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.start = 1'b1;
                weight_sel_count_clear                                      = 1'b1;
                weight_cnt_clear                                            = 1'b1;
                next_rbectrl_state                                          = RBECTRL_NORMQUANT;
              end
              else begin
                ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_accum = 1'b1;
                weight_sel_count_clear                            = 1'b1;
                next_rbectrl_state                                = RBECTRL_UPDATEIDX;
              end
            end

            else begin
              if(flags_engine_i.valid_weight_hs) begin
                weight_sel_count_enable = 1;
              end
            end

          end

        // ========================================================================
        // stream in data and compute results
        RBECTRL_NORMQUANT:
          begin

            ctrl_streamer_o.ld_which_mux_sel = LD_NORM_SEL; // select weight
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select norm

            ctrl_engine_o.ctrl_engine_core.ctrl_aq.ctrl_normquant.start = 1'b1;
            norm_cnt_en                                                 = 1'b0;
            ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_norm            = 1'b1;

            if( flags_streamer_i.norm_source_flags.ready_start  & (norm_cnt_q=='0) ) begin
              norm_cnt_en = 1'b1;
            end
            else if( (flags_streamer_i.norm_source_flags.ready_start ) & (norm_cnt_q=={'0,1'b1})) begin
              ctrl_streamer_o.norm_source_ctrl.req_start                  = 1'b1;
              norm_cnt_en                                                 = 1'b1;
            end

            if( (flags_streamer_i.norm_source_flags.ready_start) & !(empty_cycles_cnt_q=={'0,1'b1}) & (norm_cnt_q=={'0,2'b10}) ) begin
              empty_cycles_cnt_en   = 1'b1;
            end

            if (empty_cycles_cnt_q=={'0,1'b1}) begin
              nr_empty_cycles_cnt_d = nr_empty_cycles_cnt_q + {'0,1'b1};
            end

            if( (nr_empty_cycles_cnt_q == 1) & all_aq_streamout) begin
              empty_cycles_cnt_clear = 1'b1;
              norm_cnt_clear         = 1'b1;
              next_rbectrl_state     = RBECTRL_STREAMOUT;
            end

          end

        // ========================================================================
        // stream out results
        RBECTRL_STREAMOUT:
          begin
            if( (flags_streamer_i.conv_sink_flags.done | flags_streamer_i.conv_sink_flags.ready_start ) & !(conv_cnt_q=={'0,1'b1})) begin
              ctrl_streamer_o.conv_sink_ctrl.req_start = 1'b1;
              conv_cnt_en                              = 1'b1;
            end

            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b1;        // select store
            conv_cnt_en                      = 1'b0;
            if(flags_streamer_i.conv_sink_flags.done & (ctrl_i.n_tiles_qa_out=={'0,2'b10}) & (n_tiles_conv_cnt=='0) ) begin
              n_tiles_conv_cnt_en = 1'b1;
              next_rbectrl_state  = RBECTRL_STREAMOUT;
            end
            else if(flags_streamer_i.conv_sink_flags.done & flags_streamer_i.tcdm_fifo_empty) begin
              next_rbectrl_state                       = RBECTRL_UPDATEIDX;
              n_tiles_conv_cnt_clear                   = 1'b1;
              ctrl_streamer_o.conv_sink_ctrl.req_start = 1'b0;
            end
            else if(flags_streamer_i.conv_sink_flags.done & ~flags_streamer_i.tcdm_fifo_empty) begin
              next_rbectrl_state                       = RBECTRL_STREAMOUT_ENDING;
              n_tiles_conv_cnt_clear                   = 1'b1;
              ctrl_streamer_o.conv_sink_ctrl.req_start = 1'b0;
            end

            // clear tiles counter
            n_tiles_feat_cnt_clear = 1'b1;

          end

        RBECTRL_STREAMOUT_ENDING:
          begin
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b1;        // select store
            conv_cnt_en                      = 1'b0;
            if(flags_streamer_i.tcdm_fifo_empty) begin
              next_rbectrl_state                       = RBECTRL_UPDATEIDX;
              n_tiles_conv_cnt_clear                   = 1'b1;
              ctrl_streamer_o.conv_sink_ctrl.req_start = 1'b0;
            end
          end

        RBECTRL_UPDATEIDX:
          begin
            // update the indeces, then go back to load or idle
            ctrl_streamer_o.ld_which_mux_sel                  = LD_FEAT_SEL; // select weight
            ctrl_streamer_o.ld_st_mux_sel                     = 1'b1;        // select store
            ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_accum = 1'b1;

            // if loop update available store them
            if(flags_uloop_i.done | flags_uloop_i.valid) begin
              flags_uloop_last_en = 1'b1;
            end

            if(flags_uloop_i.done) begin
              next_rbectrl_state = RBECTRL_TERMINATE;
            end
            else if(flags_uloop_i.valid == 1'b0) begin
              ctrl_uloop_o.enable = 1'b1;
            end
            else if((flags_uloop_i.valid == 1'b1) & (flags_uloop_i.loop == 3'b010)) begin
              // if new round of K_OUT_TILES
              ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
              ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
              next_rbectrl_state               = RBECTRL_LOAD_FEAT;

              // prepare activation registers
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen    = 1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load = 1'b1;
              ctrl_engine_o.ctrl_engine_core.ctrl_aq.goto_accum            = 1'b1;
              // request new features
              if (flags_streamer_i.feat_source_flags.ready_start) begin
                ctrl_streamer_o.feat_source_ctrl.req_start = 1'b1;
              end
            end
            else if((flags_uloop_i.valid == 1'b1) & ((flags_uloop_i.loop == 3'b001) | (flags_uloop_i.loop == 3'b000) )) begin
              // if new round of K_IN_TILES or new round of QA_TILES
              // load again features
              ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
              ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
              next_rbectrl_state               = RBECTRL_LOAD_FEAT;

              // prepare activation registers
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen    = 1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load = 1'b1;

              // request new features
              if (flags_streamer_i.feat_source_flags.ready_start) begin
                n_tiles_feat_cnt_en                        = 1'b1;
                ctrl_streamer_o.feat_source_ctrl.req_start = 1'b1;
              end
            end
            else if((flags_uloop_i.valid == 1'b1) & ((flags_uloop_i.loop == 3'b011) | (flags_uloop_i.loop == 3'b100))) begin
              // if new round of feature tile in H/W dimension
              // load again features
              ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
              ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
              next_rbectrl_state               = RBECTRL_LOAD_FEAT;

              // prepare activation registers
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.i_vlen    = 1;
              ctrl_engine_o.ctrl_engine_core.ctrl_activation_reg.goto_load = 1'b1;

              // request new features
              if (flags_streamer_i.feat_source_flags.ready_start) begin
                // n_tiles_feat_cnt_en                        = 1'b1;
                ctrl_streamer_o.feat_source_ctrl.req_start = 1'b1;
              end
            end
            else if (flags_streamer_i.conv_sink_flags.ready_start == 1'b1) begin
              next_rbectrl_state                           = RBECTRL_TERMINATE;
            end
            else begin
              next_rbectrl_state = RBECTRL_TERMINATE;
            end
          end

        RBECTRL_WAIT:
          begin
            // wait for the flags to be ok then go back to load
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b0;        // select load
            ctrl_uloop_o.enable              = 1'b0;

            if(flags_streamer_i.feat_source_flags.ready_start & flags_streamer_i.weight_source_flags.ready_start) begin
              next_rbectrl_state = RBECTRL_LOAD_FEAT;
            end
            else if (flags_streamer_i.conv_sink_flags.ready_start == 1'b1) begin
              next_rbectrl_state = RBECTRL_TERMINATE;
            end
          end

        RBECTRL_TERMINATE:
          begin
            // wait for the flags to be ok then go back to idle
            ctrl_streamer_o.ld_which_mux_sel = LD_FEAT_SEL; // select feat
            ctrl_streamer_o.ld_st_mux_sel    = 1'b1; // select store
            if(flags_streamer_i.feat_source_flags.ready_start & flags_streamer_i.conv_sink_flags.ready_start) begin
              next_rbectrl_state = RBECTRL_IDLE;
              ctrl_slave_o.done  = 1'b1;
            end
          end

      endcase // curr_rbectrl_state
    end


  ////////////////////
  // OFFSET CONTROL //
  ////////////////////

  assign offset_state_d = ~offset_state_q;

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : accumulator_counter
    if(~rst_ni) begin
      offset_state_q <= '0;
      offset_state_delayed_0_q <= '0;
      offset_state_delayed_1_q <= '0;
      offset_state_delayed_2_q <= '0;
    end
    else begin
      if(clear_i)
        offset_state_q <= '0;
      else if(ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.offset_en == 1'b1)
        offset_state_q <= offset_state_d;

      offset_state_delayed_0_q <= offset_state_q;
      offset_state_delayed_1_q <= offset_state_delayed_0_q;
      offset_state_delayed_2_q <= offset_state_delayed_1_q;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : ws_counter
    if(~rst_ni) begin
      ws_count_q <= '0;
    end
    else begin
      if(weight_sel_count_clear)
        ws_count_q <=  '0;
      if(weight_sel_count_enable)
        ws_count_q <= ws_count_d;
    end
  end

  assign ctrl_engine_o.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.offset_en = (curr_rbectrl_state==RBECTRL_COMPUTE) ?
                                                                                        (((ws_count_q == 0 | ws_count_q == 1) & weight_sel_count_enable) ? 1'b1  :
                                                                                                                                                           1'b0) :
                                                                                                                                1'b0;

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : offset_delay
    if(~rst_ni) begin
      offset_en_delayed_0_q <= '0;
      offset_en_delayed_1_q <= '0;
      offset_en_delayed_2_q <= '0;
    end
    else begin
      offset_en_delayed_0_q <= ((ws_count_q == 0 | ws_count_q == 1) & weight_sel_count_enable) ? 1'b1: 1'b0;
      offset_en_delayed_1_q <= offset_en_delayed_0_q;
      offset_en_delayed_2_q <= offset_en_delayed_1_q;
    end
  end


  //////////////////////////
  // EMPTY CYCLES COUNTER //
  //////////////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      nr_empty_cycles_cnt_q    <= '0;
    end
    else begin
      if (empty_cycles_cnt_clear | clear_i) begin
        nr_empty_cycles_cnt_q    <= '0;
      end
      else begin
        nr_empty_cycles_cnt_q <= nr_empty_cycles_cnt_d;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni)
  begin
    if(~rst_ni) begin
      empty_cycles_cnt_q <= '0;
    end
    else begin
      if (empty_cycles_cnt_clear | clear_i) begin
        empty_cycles_cnt_q <= '0;
      end
      if (empty_cycles_cnt_en) begin
        empty_cycles_cnt_q <= empty_cycles_cnt_q + 1;
      end
    end
  end


  //////////////////
  // CONV COUNTER //
  //////////////////
  always_ff @(posedge clk_i or negedge rst_ni)
  begin : conv_cnt
    if(~rst_ni) begin
      conv_cnt_q <= '0;
    end
    else begin
      if (conv_cnt_clear | clear_i) begin
        conv_cnt_q <= '0;
      end
      if (conv_cnt_en) begin
        conv_cnt_q <= conv_cnt_q + 1;
      end
    end
  end


  //////////////////
  // NORM COUNTER //
  //////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : weight_cnt
    if(~rst_ni) begin
      norm_cnt_q    <= '0;
    end
    else begin
      if (norm_cnt_clear | clear_i) begin
        norm_cnt_q    <= '0;
      end
      if (norm_cnt_en) begin
        norm_cnt_q <= norm_cnt_q + 1;
      end
    end
  end


  ////////////////////
  // WEIGHT COUNTER //
  ////////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
  begin : norm_cnt
    if(~rst_ni) begin
      weight_cnt_q    <= '0;
    end
    else begin
      if (weight_cnt_clear | clear_i) begin
        weight_cnt_q    <= '0;
      end
      if (weight_cnt_en) begin
        weight_cnt_q <= weight_cnt_q + 1;
      end
    end
  end


  //////////////////////////
  // FEATURE TILE COUNTER //
  //////////////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
    begin : feature_tile_counter
      if(~rst_ni) begin
          n_tiles_feat_cnt  <= '0;
      end
      else begin
        if(n_tiles_feat_cnt_clear | clear_i) begin
          n_tiles_feat_cnt  <= '0;
        end
        else if(n_tiles_feat_cnt_en) begin
          n_tiles_feat_cnt  <= n_tiles_feat_cnt + 1;
        end
      end
    end


  ///////////////////////
  // CONV TILE COUNTER //
  ///////////////////////

  always_ff @(posedge clk_i or negedge rst_ni)
    begin : conv_tile_counter
      if(~rst_ni) begin
          n_tiles_conv_cnt  <= '0;
      end
      else begin
        if(n_tiles_conv_cnt_clear | clear_i) begin
          n_tiles_conv_cnt  <= '0;
        end
        else if(n_tiles_conv_cnt_en) begin
          n_tiles_conv_cnt  <= n_tiles_conv_cnt + 1;
        end
      end
    end


  /////////////////////////
  // ULOOP FLAGS STORING //
  /////////////////////////
  always_ff @(posedge clk_i or negedge rst_ni)
    begin : flags_uloop_last
      if(~rst_ni) begin
          flags_uloop_valid_loop <= '0;
          flags_uloop_done_loop  <= '0;
          flags_uloop_last_loop  <= '0;
      end
      else begin
        if(flags_uloop_last_clear | clear_i) begin
          flags_uloop_valid_loop <= '0;
          flags_uloop_done_loop  <= '0;
          flags_uloop_last_loop  <= '0;
        end
        else if(flags_uloop_last_en) begin
          flags_uloop_valid_loop <= flags_uloop_i.valid;
          flags_uloop_done_loop  <= flags_uloop_i.done;
          flags_uloop_last_loop  <= flags_uloop_i.loop;
        end
      end
    end

endmodule // rbe_ctrl

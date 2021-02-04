/*
 * rbe_engine.sv
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


module rbe_engine
  import rbe_package::*;
#(
  parameter  int unsigned BW                  = 32*9,
  localparam int unsigned RBE_LD_SIZE         = 9,
  localparam int unsigned BC_COLUMN_SIZE      = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv blocks per column (default 9)
  localparam int unsigned BC_NR_COLUMN        = rbe_package::BINCONV_NR_COLUMN,   // number of BinConv columns (default 9 -- same of size of BinConv columns!)
  localparam int unsigned BC_BLOCK_SIZE       = rbe_package::BINCONV_BLOCK_SIZE,  // number of SoP's per BinConv block (default 4)
  localparam int unsigned BC_FEAT_BUFFER_SIZE = rbe_package::BINCONV_FEAT_BUFFER_SIZE,
  localparam int unsigned BC_NR_ACTIVATIONS   = BC_FEAT_BUFFER_SIZE*BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE,
  localparam int unsigned TP                  = rbe_package::BINCONV_TP,          // number of input elements processed per cycle
  localparam int unsigned NR_A_STREAMS_MAX    = BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE
) (
  // global signals
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   test_mode_i,
  // local enable & clear
  input  logic                   enable_i,
  input  logic                   clear_i,
  // input streams + handshake
  hwpe_stream_intf_stream.sink   feat_i,
  hwpe_stream_intf_stream.sink   weight_i,
  hwpe_stream_intf_stream.sink   norm_i,
  hwpe_stream_intf_stream.source conv_o,

  input  ctrl_engine_t           ctrl_i,
  output flags_engine_t          flags_o
);

  ////////////////////////////
  // INTERFACES AND SIGNALS //
  ////////////////////////////

  // weight stream
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) weight_fifo (.clk(clk_i));

  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) load_weight          [RBE_LD_SIZE-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) load_weight_silenced [RBE_LD_SIZE-1:0] (.clk(clk_i));

  // activation/feature streams
  logic [TP*BC_FEAT_BUFFER_SIZE-1:0] a_used_data;

  hwpe_stream_intf_stream #(.DATA_WIDTH(TP*BC_FEAT_BUFFER_SIZE)) a_used                           (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(TP*BC_FEAT_BUFFER_SIZE)) act_demux [NR_A_STREAMS_MAX-1:0] (.clk(clk_i));

  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) a_x_pixels              [              BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) a_x_pixels_masked_fs    [              BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) a_x_pixels_bitidx_demux [BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE-1:0] (.clk(clk_i));

  hwpe_stream_intf_stream #(.DATA_WIDTH(TP))                  a_x_col_fenced     [   NR_A_STREAMS_MAX-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(NR_A_STREAMS_MAX*TP)) a_x_col                                      (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(NR_A_STREAMS_MAX*TP)) a_x_col_demux_rows [BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(NR_A_STREAMS_MAX*TP)) a_y_rows_masked_fs [BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(NR_A_STREAMS_MAX*TP)) a_y_rows_fenced    [BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));

  hwpe_stream_intf_stream #(.DATA_WIDTH(BC_FEAT_BUFFER_SIZE*NR_A_STREAMS_MAX*TP )) a_x_y (.clk(clk_i));

  hwpe_stream_intf_stream #(.DATA_WIDTH(TP)) activation_flatten [NR_A_STREAMS_MAX*BC_FEAT_BUFFER_SIZE-1:0] (.clk(clk_i));

  // normalization stream
  hwpe_stream_intf_stream #(.DATA_WIDTH(BW)) norm_fifo (.clk(clk_i));

  // convolution results stream
  hwpe_stream_intf_stream #(.DATA_WIDTH(TP))   store_out        [BC_NR_COLUMN-1:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(3*TP)) conv_merged      [             2:0] (.clk(clk_i));
  hwpe_stream_intf_stream #(.DATA_WIDTH(3*TP)) conv_merged_fifo [             2:0] (.clk(clk_i));

  logic [      $clog2(BC_BLOCK_SIZE)-1:0] sel_a_bitidx_demux;
  logic [$clog2(BC_FEAT_BUFFER_SIZE)-1:0] sel_a_pixelrow_demux;
  logic [      $clog2(BC_BLOCK_SIZE)-1:0] curr_a_bitidx_cnt, next_a_bitidx_cnt;
  logic [$clog2(BC_FEAT_BUFFER_SIZE)-1:0] curr_a_pixelrow_cnt, next_a_pixelrow_cnt;

  logic [              (3*TP)-1:0] conv_data;
  logic [    (RBE_LD_SIZE*TP)-1:0] conv_data_extended;
  logic [          ((3*TP)/8)-1:0] conv_strb;
  logic [((RBE_LD_SIZE*TP)/8)-1:0] conv_strb_extended;
  logic                            conv_valid;
  logic                            conv_ready;

  logic [2:0] conv_hs;
  logic [1:0] c_out_cnt_q, c_out_cnt_d;
  logic [1:0] sel_c_out_q, sel_c_out_d;
  logic [1:0] sel_c_out_pixel;


  /////////////////
  // WEIGHT DATA //
  /////////////////
  /* Weight dataflow
   *                   i_w_fifo              |--> load_weight[0] --> load_weight_silenced[0]
   *                   ---------             |         .                     .
   * weight_i(9*32) -->|   |   |--> i_w_split|         .                     .
   *                   ---------             |         .                     .
   *                                         |--> load_weight[8] --> load_weight_silenced[8]
   */

  // FIFO
  hwpe_stream_fifo #(
    .DATA_WIDTH           ( BW ),
    .FIFO_DEPTH           ( 2  ),
    .LATCH_FIFO           ( 0  ),
    .LATCH_FIFO_TEST_WRAP ( 0  )
  ) i_w_fifo (
    .clk_i   ( clk_i       ),
    .rst_ni  ( rst_ni      ),
    .clear_i ( clear_i     ),
    .flags_o (             ),
    .push_i  ( weight_i    ),
    .pop_o   ( weight_fifo )
  );

  // split weights into 9 streams a 32bit data
  hwpe_stream_split #(
    .NB_OUT_STREAMS ( RBE_LD_SIZE ),
    .DATA_WIDTH_IN  ( BW          )
  ) i_w_split (
    .clk_i   ( clk_i       ),
    .rst_ni  ( rst_ni      ),
    .clear_i ( clear_i     ),
    .push_i  ( weight_fifo ),
    .pop_o   ( load_weight )
  );
  // asign flags
  assign flags_o.valid_weight_hs = weight_fifo.valid & weight_fifo.ready;

  // silence weights
  for(genvar c=0; c<RBE_LD_SIZE; c++) begin : column_silencing
    // silence data, strb signals
    assign load_weight_silenced[c].data = ctrl_i.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.row_onehot_en[c] ? load_weight[c].data  : '0;
    assign load_weight_silenced[c].strb = ctrl_i.ctrl_engine_core.ctrl_array.ctrl_column.ctrl_block.row_onehot_en[c] ? load_weight[c].strb  : '0;
    // let ready and valid propagate
    assign load_weight[c].ready          = load_weight_silenced[c].ready;
    assign load_weight_silenced[c].valid = load_weight[c].valid;
  end


  ////////////////////////
  // NORMALIZATION DATA //
  ////////////////////////
  /* Normalization dataflow
   *            i_w_fifo
   *           ---------
   * norm_i -->|   |   |--> norm_fifo
   *           ---------
   */

  hwpe_stream_fifo #(
    .DATA_WIDTH           ( BW ),
    .FIFO_DEPTH           ( 2  ),
    .LATCH_FIFO           ( 0  ),
    .LATCH_FIFO_TEST_WRAP ( 0  )
  ) i_n_fifo (
    .clk_i   ( clk_i     ),
    .rst_ni  ( rst_ni    ),
    .clear_i ( clear_i   ),
    .flags_o (           ),
    .push_i  ( norm_i    ),
    .pop_o   ( norm_fifo )
  );

  /////////////////
  // CONVOLUTION //
  /////////////////
  /* Normalization dataflow
   *
   * store_out[0]|                    -------
   * store_out[1]|-->conv_merged[0]-->|  |  |-->conv_merged_fifo[0]
   * store_out[2]|                    -------
   *
   * store_out[3]|                    -------
   * store_out[4]|-->conv_merged[1]-->|  |  |-->conv_merged_fifo[1]
   * store_out[5]|                    -------
   *
   * store_out[6]|                    -------
   * store_out[7]|-->conv_merged[2]-->|  |  |-->conv_merged_fifo[2]
   * store_out[8]|                    -------
   *
   */


  for (genvar ii=0; ii<3; ii++) begin : conv_merge_gen
      // always merge the streams of 3 columns (their accumulator-normquant output)
      hwpe_stream_merge #(
        .NB_IN_STREAMS ( 3  ),
        .DATA_WIDTH_IN ( TP )
      ) i_c_merge
      (
        .clk_i   ( clk_i                      ),
        .rst_ni  ( rst_ni                     ),
        .clear_i ( clear_i                    ),
        .push_i  ( store_out [(ii*3+2):ii*3 ] ),
        .pop_o   ( conv_merged [ii]           )
      );

        hwpe_stream_fifo #(
        .DATA_WIDTH           ( TP*3 ),
        .FIFO_DEPTH           ( 2    ),
        .LATCH_FIFO           ( 0    ),
        .LATCH_FIFO_TEST_WRAP ( 0    )
      ) i_c_fifo (
        .clk_i   ( clk_i                 ),
        .rst_ni  ( rst_ni                ),
        .clear_i ( clear_i               ),
        .flags_o (                       ),
        .push_i  ( conv_merged [ii]      ),
        .pop_o   ( conv_merged_fifo [ii] )
      );

      assign conv_hs[ii] = conv_merged_fifo[ii].valid & conv_merged_fifo[ii].ready;

  end // accumulator_gen

  assign c_out_cnt_d = (c_out_cnt_q == 2'b11)? '0 : c_out_cnt_q + 1;

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        c_out_cnt_q <= 1'b0;
      else if(conv_hs[0] | conv_hs[1] | conv_hs[2])
        c_out_cnt_q <= c_out_cnt_d;
    end

  assign sel_c_out_d = (sel_c_out_q == 2'b10)? '0 : sel_c_out_q+1;

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        sel_c_out_q <= 1'b0;
      else if((c_out_cnt_q == 2'b11) & (conv_hs[0] | conv_hs[1] | conv_hs[2]))
        sel_c_out_q <= sel_c_out_d;
    end

  assign sel_c_out_pixel = sel_c_out_q;

  // tcdm ports binding
  assign conv_valid = (sel_c_out_pixel[0]) ? conv_merged_fifo[1].valid :
                      (sel_c_out_pixel[1]) ? conv_merged_fifo[2].valid :
                                             conv_merged_fifo[0].valid;

  assign conv_data  = (sel_c_out_pixel[0]) ? conv_merged_fifo[1].data  :
                      (sel_c_out_pixel[1]) ? conv_merged_fifo[2].data  :
                                             conv_merged_fifo[0].data;

  assign conv_strb  = (sel_c_out_pixel[0]) ? conv_merged_fifo[1].strb  :
                      (sel_c_out_pixel[1]) ? conv_merged_fifo[2].strb  :
                                             conv_merged_fifo[0].strb;


  assign conv_data_extended = { {((RBE_LD_SIZE-3)*TP){1'b0}}, conv_data };
  assign conv_strb_extended = { {(((RBE_LD_SIZE-3)*TP)/8){1'b0}}, conv_strb };

  assign conv_merged_fifo[0].ready = (!sel_c_out_pixel[1] & !sel_c_out_pixel[0]) ? conv_ready : 1'b0;
  assign conv_merged_fifo[1].ready = (!sel_c_out_pixel[1] &  sel_c_out_pixel[0]) ? conv_ready : 1'b0;
  assign conv_merged_fifo[2].ready = ( sel_c_out_pixel[1] & !sel_c_out_pixel[0]) ? conv_ready : 1'b0;

  assign conv_o.data  = conv_data_extended;
  assign conv_o.strb  = conv_strb_extended;
  assign conv_o.valid = conv_valid;
  assign conv_ready   = conv_o.ready;


  ////////////////////
  // ACTIVATION DATA//
  ////////////////////
  /* Activation dataflow
   *
   * 1) Take 3 or 5 pixels in x-direction (rest is silenced)
   *                       i_a_pixel_split
   * feat_i(9*32) --> a_used --> | a_x_pixels[0]--> a_x_pixels_masked_fs[0]
   *          (take 5*32)        | a_x_pixels[1]--> a_x_pixels_masked_fs[1]
   *                             | a_x_pixels[2]--> a_x_pixels_masked_fs[2]
   *                             | a_x_pixels[3] -> a_x_pixels_masked_fs[3]
   *                             | a_x_pixels[4]--> a_x_pixels_masked_fs[4]
   *                                            (if FS=1: idx 3,4 are silenced)
   *
   * 2) take 4 each (demultiplexed over time)                  i_a_fence               i_a_merge
   *                             /|-->a_x_pixels_bitidx_demux[0] -->|-->a_x_col_fenced[0] -->|
   *                            | |-->a_x_pixels_bitidx_demux[1] -->|-->a_x_col_fenced[1] -->|
   *  a_x_pixels_masked_fs[0]-->| |-->a_x_pixels_bitidx_demux[2] -->|-->a_x_col_fenced[2] -->|
   *           ...               \|-->a_x_pixels_bitidx_demux[3] -->|-->a_x_col_fenced[3] -->|
   *           ...              ...                ...              |                        |--> a_x_col
   *           ...               /|-->a_x_pixels_bitidx_demux[16]-->|-->a_x_col_fenced[16]-->|
   *  a_x_pixels_masked_fs[4]-->| |-->a_x_pixels_bitidx_demux[17]-->|-->a_x_col_fenced[17]-->|
   *                            | |-->a_x_pixels_bitidx_demux[18]-->|-->a_x_col_fenced[18]-->|
   *                             \|-->a_x_pixels_bitidx_demux[19]-->|-->a_x_col_fenced[19]-->|
   *
   * 3) demultiplex over time into 3 or 5 pixel rows in y-direction
   *
   *             /|-->a_x_col_demux_rows[0]-->a_y_rows_masked_fs[0]
   *            | |-->a_x_col_demux_rows[1]-->a_y_rows_masked_fs[1]
   *  a_x_col-->| |-->a_x_col_demux_rows[2]-->a_y_rows_masked_fs[2]
   *            | |-->a_x_col_demux_rows[3]-->a_y_rows_masked_fs[3]
   *             \|-->a_x_col_demux_rows[4]-->a_y_rows_masked_fs[4]
   *
   *                        i_a_fence          i_a_merge_2
   *  a_y_rows_masked_fs[0]-->|--> a_y_rows_fenced|
   *  a_y_rows_masked_fs[1]-->|--> a_y_rows_fenced|
   *  a_y_rows_masked_fs[2]-->|--> a_y_rows_fenced|-->a_x_y
   *  a_y_rows_masked_fs[3]-->|--> a_y_rows_fenced|
   *  a_y_rows_masked_fs[4]-->|--> a_y_rows_fenced|
   *
   *          |--> activation_flatten[0]  -->
   *          |--> activation_flatten[1]  -->
   *  a_x_y-->|            ...            -->  Feed into accelerator
   *          |            ...            -->
   *          |--> activation_flatten[98] -->
   *
   */

  // only use the 5x32 bit data
  // silence the other 4x32 bit data
  for(genvar jj=0; jj<TP*BC_FEAT_BUFFER_SIZE; jj++) begin : a_used_part_gen
    assign a_used_data[jj] = feat_i.data[jj];
  end
  assign a_used.data  = a_used_data;
  assign a_used.valid = feat_i.valid;
  assign a_used.strb  = feat_i.strb;
  assign feat_i.ready = a_used.ready;

  // split incoming data of 5x32bit into its 5 pixels of each 32bit
  hwpe_stream_split #(
    .NB_OUT_STREAMS ( BC_FEAT_BUFFER_SIZE    ),
    .DATA_WIDTH_IN  ( BC_FEAT_BUFFER_SIZE*TP )
  ) i_a_pixel_split (
    .clk_i   ( clk_i         ),
    .rst_ni  ( rst_ni        ),
    .clear_i ( clear_i       ),
    .push_i  ( a_used        ),
    .pop_o   ( a_x_pixels    )
  );

  // if FS=1: take a pixel set of 3x3 -> silence 2 out of 5 pixel columns
  // if FS=3: take a pixel set of 5x5 -> take all 5 pixel columns
  for(genvar p=0; p<BC_FEAT_BUFFER_SIZE; p++) begin : pixel_x_silencing

    if(p<3) begin // take the pixels anyway
      assign a_x_pixels_masked_fs[p].data  = a_x_pixels[p].data;
      assign a_x_pixels_masked_fs[p].strb  = a_x_pixels[p].strb;

      assign a_x_pixels[p].ready           = a_x_pixels_masked_fs[p].ready;
      assign a_x_pixels_masked_fs[p].valid = a_x_pixels[p].valid;
    end
    else begin // silence unused pixels if FS=1
      // silence data, strb signals
      assign a_x_pixels_masked_fs[p].data  = (ctrl_i.fs == 3) ? a_x_pixels[p].data : '0;
      assign a_x_pixels_masked_fs[p].strb  = (ctrl_i.fs == 3) ? a_x_pixels[p].strb : '0;

      // let ready and valid propagate
      assign a_x_pixels[p].ready           = a_x_pixels_masked_fs[p].ready;
      assign a_x_pixels_masked_fs[p].valid = a_x_pixels[p].valid; // (ctrl_i.fs == 3) ? a_x_pixels[p].valid : 1'b1;
    end
  end

  // demux each pixel into
  for(genvar pp=0; pp<BC_FEAT_BUFFER_SIZE; pp++) begin : a_pixel_demux
      // demux each pixel into it's 4bit indexes (over time)
      hwpe_stream_demux_static #(
        .NB_OUT_STREAMS ( BC_BLOCK_SIZE )
      ) i_a_demux (
        .clk_i   ( clk_i                                                                ),
        .rst_ni  ( rst_ni                                                               ),
        .sel_i   ( sel_a_bitidx_demux                                                     ),
        .clear_i ( clear_i                                                              ),
        .push_i  ( a_x_pixels_masked_fs[pp]                                             ),
        .pop_o   ( a_x_pixels_bitidx_demux[(pp+1)*BC_BLOCK_SIZE-1 : (pp*BC_BLOCK_SIZE)] )
      );
  end

  // wait until all activations are demuxed (over time)
  // and all 4 blocks of this pixel row is loaded
  hwpe_stream_fence #(
    .NB_STREAMS ( BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE ),
    .DATA_WIDTH ( TP                                )
  ) i_a_fence (
    .clk_i       ( clk_i                   ),
    .rst_ni      ( rst_ni                  ),
    .clear_i     ( clear_i                 ),
    .test_mode_i ( test_mode_i             ),
    .push_i      ( a_x_pixels_bitidx_demux ),
    .pop_o       ( a_x_col_fenced          )
  );

  // merge the pixel row of 5x4x32 bit activations
  hwpe_stream_merge #(
    .NB_IN_STREAMS ( BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE ),
    .DATA_WIDTH_IN ( TP                                )
  ) i_a_merge
  (
    .clk_i   ( clk_i          ),
    .rst_ni  ( rst_ni         ),
    .clear_i ( clear_i        ),
    .push_i  ( a_x_col_fenced ),
    .pop_o   ( a_x_col        )
  );

  // demux the pixel rows in y-direction
  // if FS==1: demuxing over three pixel rows
  // if FS==3: demuxing over all five pixel rows
  hwpe_stream_demux_static #(
    .NB_OUT_STREAMS ( BC_FEAT_BUFFER_SIZE )
  ) i_a_demux2 (
    .clk_i   ( clk_i              ),
    .rst_ni  ( rst_ni             ),
    .sel_i   ( sel_a_pixelrow_demux    ),
    .clear_i ( clear_i            ),
    .push_i  ( a_x_col            ),
    .pop_o   ( a_x_col_demux_rows )
  );

  // Silence 2 out of 5 pixel rows if FS==1 is running
  for(genvar p=0; p<BC_FEAT_BUFFER_SIZE; p++) begin : pixel_y_silencing

    if(p<3) begin
      assign a_y_rows_masked_fs[p].data  = a_x_col_demux_rows[p].data;
      assign a_y_rows_masked_fs[p].strb  = a_x_col_demux_rows[p].strb;

      assign a_x_col_demux_rows[p].ready = a_y_rows_masked_fs[p].ready;
      assign a_y_rows_masked_fs[p].valid = a_x_col_demux_rows[p].valid;
    end
    else begin
      // silence data, strb signals
      assign a_y_rows_masked_fs[p].data = (ctrl_i.fs == 3) ? a_x_col_demux_rows[p].data : '0;
      assign a_y_rows_masked_fs[p].strb = (ctrl_i.fs == 3) ? a_x_col_demux_rows[p].strb : '0;

      // let ready and valid propagate
      assign a_x_col_demux_rows[p].ready = a_y_rows_masked_fs[p].ready;
      assign a_y_rows_masked_fs[p].valid = (ctrl_i.fs == 3) ? a_x_col_demux_rows[p].valid : a_y_rows_masked_fs[p].ready;
    end
  end

  // wait until all 5 pixel rows have arrived (5 pixel rows or 3 plus 2 silenced ones)
  hwpe_stream_fence #(
    .NB_STREAMS ( BC_FEAT_BUFFER_SIZE                  ),
    .DATA_WIDTH ( BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE*TP )
  ) i_a_fence_2 (
    .clk_i       ( clk_i              ),
    .rst_ni      ( rst_ni             ),
    .clear_i     ( clear_i            ),
    .test_mode_i ( test_mode_i        ),
    .push_i      ( a_y_rows_masked_fs ),
    .pop_o       ( a_y_rows_fenced    )
  );

  // merge all demuxed rows and columns
  hwpe_stream_merge #(
    .NB_IN_STREAMS ( BC_FEAT_BUFFER_SIZE                  ),
    .DATA_WIDTH_IN ( BC_FEAT_BUFFER_SIZE*BC_BLOCK_SIZE*TP )
  ) i_a_merge_2
  (
    .clk_i   ( clk_i           ),
    .rst_ni  ( rst_ni          ),
    .clear_i ( clear_i         ),
    .push_i  ( a_y_rows_fenced ),
    .pop_o   ( a_x_y           )
  );

  // assign helper flag
  assign flags_o.feat_fence_arrived = a_x_y.valid;

  // split the activation into 100 data streams of 32
  hwpe_stream_split #(
    .NB_OUT_STREAMS ( BC_NR_ACTIVATIONS    ),
    .DATA_WIDTH_IN  ( BC_NR_ACTIVATIONS*TP )
  ) i_a_split (
    .clk_i   ( clk_i              ),
    .rst_ni  ( rst_ni             ),
    .clear_i ( clear_i            ),
    .push_i  ( a_x_y              ),
    .pop_o   ( activation_flatten )
  );


  //////////////////////////////
  // Activation Demux Control //
  //////////////////////////////

  // activation pixel row counter
  assign next_a_pixelrow_cnt = (ctrl_i.fs==3) ? ((curr_a_pixelrow_cnt==BC_FEAT_BUFFER_SIZE-1)   ? '0 : curr_a_pixelrow_cnt + 1) :
                                                ((curr_a_pixelrow_cnt==BUFFER_PATCH_SIZE_MIN-1) ? '0 : curr_a_pixelrow_cnt + 1);

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        curr_a_pixelrow_cnt <= 1'b0;
      else if(feat_i.valid & feat_i.ready & (curr_a_bitidx_cnt==BC_BLOCK_SIZE-1))
        curr_a_pixelrow_cnt <= next_a_pixelrow_cnt;
    end

  // activation bitindex counter
  assign next_a_bitidx_cnt = (curr_a_bitidx_cnt==BC_BLOCK_SIZE-1) ? '0 : curr_a_bitidx_cnt + 1;

  always_ff @(posedge clk_i or negedge rst_ni)
    begin
      if(~rst_ni)
        curr_a_bitidx_cnt <= 1'b0;
      else if(feat_i.valid & feat_i.ready)
        curr_a_bitidx_cnt <= next_a_bitidx_cnt;
    end

  // assign counters to demux select signals
  assign sel_a_bitidx_demux   = curr_a_bitidx_cnt;
  assign sel_a_pixelrow_demux = curr_a_pixelrow_cnt;

  // set flags
  assign flags_o.sel_pixel_demux  = sel_a_pixelrow_demux;
  assign flags_o.sel_bitidx_demux = sel_a_bitidx_demux;
  assign flags_o.sel_c_out_pixel  = sel_c_out_pixel;


  ////////////////
  // RBE Engine //
  ////////////////

  rbe_engine_core #(
    .RBE_LD_SIZE       ( RBE_LD_SIZE       ),
    .BC_COLUMN_SIZE    ( BC_COLUMN_SIZE    ),
    .BC_NR_COLUMN      ( BC_NR_COLUMN      ),
    .BC_BLOCK_SIZE     ( BC_BLOCK_SIZE     ),
    .BC_NR_ACTIVATIONS ( BC_NR_ACTIVATIONS ),
    .TP                ( TP                )
  ) i_engine_core (
    .clk_i                ( clk_i                     ),
    .rst_ni               ( rst_ni                    ),
    .test_mode_i          ( test_mode_i               ),
    .enable_i             ( enable_i                  ),
    .clear_i              ( clear_i                   ),
    .activation_flatten_i ( activation_flatten        ),
    .weight_i             ( load_weight_silenced      ),
    .norm_i               ( norm_fifo                 ),
    .conv_o               ( store_out                 ),
    .ctrl_i               ( ctrl_i.ctrl_engine_core   ),
    .flags_o              ( flags_o.flags_engine_core )
  );

endmodule // rbe_engine

def print_args(args,logger):
    logger.info("--" + "Basic Config" + "--")
    logger.info(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    logger.info(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    logger.info('')

    logger.info("--" + "Data Loader" + "--")
    logger.info(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    logger.info(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    logger.info(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    logger.info(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    logger.info('')

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        logger.info("--" + "Forecasting Task" + "--")
        logger.info(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        logger.info(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        logger.info(f'  {"Inverse:":<20}{args.inverse:<20}')
        logger.info('')

    if args.task_name == 'imputation':
        logger.info("--" + "Imputation Task" + "--")
        logger.info(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        logger.info('')

    if args.task_name == 'anomaly_detection':
        logger.info("--" + "Anomaly Detection Task" + "--")
        logger.info(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        logger.info('')

    logger.info("--" + "Model Parameters" + "--")
    logger.info(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    logger.info(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    logger.info(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    logger.info(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    logger.info(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    logger.info(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    logger.info(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    logger.info(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    logger.info(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    logger.info('')
    logger.info(f'  {"Use_gcn:":<20}{args.use_gcn:<20}{"Node_dim:":<20}{args.node_dim:<20}')
    logger.info(f'  {"Gcn_depth:":<20}{args.gcn_depth:<20}{"Gcn_dropout:":<20}{args.gcn_dropout:<20}')
    logger.info(f'  {"Propalpha:":<20}{args.propalpha:<20}{"Propbeta:":<20}{args.propbeta:<20}')
    logger.info(f'  {"adj_data:":<20}{args.adj_data:<20}{"has_adj:":<20}{args.has_adj:<20}')
    logger.info(f'  {"seasonMix_type:":<20}{args.seasonMix_type:<20}{"trendMix_type:":<20}{args.trendMix_type:<20}')

    logger.info("--" + "Run Parameters" + "--")
    logger.info(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    logger.info(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    logger.info(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    logger.info(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    logger.info(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    logger.info('')

    logger.info("--" + "GPU" + "--")
    logger.info(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    logger.info('')

    logger.info("--" + "De-stationary Projector Params" + "--")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    logger.info(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    logger.info('')

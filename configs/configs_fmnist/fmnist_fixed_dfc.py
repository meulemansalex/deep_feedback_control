config = {
'alpha_di': 0.7985041176538932,
'epsilon': 1.4648080760625163e-10,
'lr': 5.423883572845198e-05,
'dataset': 'fashion_mnist',
'num_train': 1000,
'num_test': 1000,
'num_val': 1000,
'no_preprocessing_mnist': False,
'no_val_set': False,
'epochs': 100,
'batch_size': 128,
'lr_fb': 0.1,
'lr_fb_init': 0.1,
'target_stepsize': 0.1,
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'sigma': 0.15,
'forward_wd': 0,
'feedback_wd': 0,
'train_parallel': False,
'normalize_lr': True,
'epochs_fb': 0,
'freeze_forward_weights': False,
'freeze_fb_weights': True,
'freeze_fb_weights_output': False,
'shallow_training': False,
'extra_fb_epochs': 2,
'extra_fb_minibatches': 0,
'only_train_first_layer': False,
'train_only_feedback_parameters': False,
'clip_grad_norm': 1.0,
'grad_deltav_cont': True,
'beta1': 0.99,
'beta2': 0.99,
'beta1_fb': 0.99,
'beta2_fb': 0.99,
'epsilon_fb': 1e-08,
'num_hidden': 3,
'size_hidden': 256,
'size_input': 784,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'network_type': 'DFC',
'initialization': 'xavier_normal',
'fb_activation': 'tanh',
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'hpsearch': False,
'multiple_hpsearch': False,
'single_precision': False,
'evaluate': False,
'out_dir': None,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GNT_angle': False,
'save_condition_gn': False,
'save_df': False,
'gn_damping': 0.0,
'log_interval': 30,
'gn_damping_hpsearch': False,
'save_nullspace_norm_ratio': False,
'save_fb_statistics_init': False,
'compute_gn_condition_init': False,
'ndi': False,
'dt_di': 0.001,
'dt_di_fb': 0.01,
'tmax_di': 1500,
'tmax_di_fb': 100,
'epsilon_di': 0.5,
'reset_K': False,
'initialization_K': 'weight_product',
'noise_K': 0.0,
'compare_with_ndi': False,
'learning_rule': 'nonlinear_difference',
'use_initial_activations': False,
'c_homeostatic': -1,
'k_p': 2.0,
'k_p_fb': 0.0,
'inst_system_dynamics': False,
'alpha_fb': 1.0,
'noisy_dynamics': False,
'fb_learning_rule': 'normal_controller',
'inst_transmission': True,
'inst_transmission_fb': True,
'time_constant_ratio': 0.2,
'time_constant_ratio_fb': 0.005,
'apical_time_constant': -1,
'apical_time_constant_fb': 0.1,
'use_homeostatic_wd_fb': False,
'efficient_controller': False,
'proactive_controller': True,
'simulate_layerwise': True,
'include_non_converged_samples': True,
}

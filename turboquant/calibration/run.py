def calibrate_quantizer(pipeline, samples):
    """
    Calibrate quantizers using real data samples.
    """
    for x in samples:
        # Fit keys
        pipeline._get_k_quant().fit(x)
        # Fit values
        pipeline._get_v_quant().fit(x)

import pandas as pd

def save_results(mean_metrics, std_metrics, config_var, base_name):
    collection = [
        mean_metrics['accuracy'],
        std_metrics['accuracy'],
        mean_metrics['f1_score'],
        std_metrics['f1_score'],
        mean_metrics['precision'],
        std_metrics['precision'],
        mean_metrics['recall'],
        std_metrics['recall'],
        config_var['sample_factor'],
        config_var['augment_factor'],
        config_var['vae'],
        config_var['cnn'],
        config_var['lat_dim']
    ]
    index = [
        'accuracy', 'std_accuracy', 'f1_score', 'std_f1_score', 'precision',
        'std_precision', 'recall', 'std_recall',
        'sample_factor', 'augment_factor', 'vae', 'cnn', 'lat_dim'
    ]
    df = pd.DataFrame(dict(zip(index, collection)))
    df.to_pickle(f'/Users/charlherbst/generative_data_augmentation_gibbons/results/{base_name}.pkl')
    return df






Dataset Repo : 

  [https://zenodo.org/me/uploads?q=&f=shared_with_me%3Afalse&l=list&p=1&s=10&sort=newest](https://zenodo.org/records/15497355)

Experimental Setup and Hyperparameters: 

    The experiments were performed on a system equipped with an NVIDIA GeForce RTX 2080 Ti configuration featuring four GPUs, each with 11 GB of dedicated memory. 
    
    For the training experiments with seq2seq, we configured a training and evaluation batch size of 2 with a learning rate of $1 \times 10^{-5}$. 
    
    A warm-up learning rate scheduler is also used with warm-up steps set to 500, considering the smaller dataset size and small pretrained models. 
    
    The finetuning procedure is employed with a total of 3 epochs, based on which the final evaluation is made.
    
    To build the Balti ASR model from scratch, we employed the Kaldi-based training setup, leveraging acoustic feature modeling using GMMs and aligning them with HMM states. 
    
    Linear discriminant analysis (LDA) and maximum likelihood linear transform (MLLT) are applied to normalize the speaker variability. 
    
    The generated alignments in GMM-HMM are adapted with TDNNs to efficiently model long-range dependencies. 
    
    For fine-tuning, we utilized various pre-trained Whisper models with different parameter sizes, and trained them on multilingual datasets containing varying volumes of language data. 
    
    Since Balti is written in the Perso-Arabic script, similar to Urdu, we chose Urdu as the foundation for fine-tuning. 
    
    While the smaller pre-trained models were not trained on extensive Urdu datasets, the Whisper large model is not utilized due to computational constraints. 
    

Evaluation Results:: 
  
    PreTrained Model: /data3/sharif/Datasets/openai_whisper_base
    
    Dataset: /data3/sharif/Datasets/BaltiSpeechDataset

    WER: 10.29%
        CER: 3.98%


Sample Predictions and References:

    Reference : ہرژیمہ
    Prediction: تینگمہ
    
    Reference : بوہمہ
    Prediction: بوہمہ
    
    Reference : رگو
    Prediction: رگو
    
    Reference : چھیمہ
    Prediction: چھیمہ
    
    Reference : نیلم
    Prediction: نیلم
    
    Reference : ربیا
    Prediction: ربیا

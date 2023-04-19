import keras
import keras_nlp
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split


from tensorflow.keras import mixed_precision

def MLM_Continued_Pretraining(InputTextList:list, modelSavePath:str, trainingEpochs:1,
                               testSize=0.1,baseModel='bert_tiny_en_uncased', sequenceLength=512, 
                               maskSelectionRate=0.25, maskSelectionLength=64, batchSize=64,
                               learningRate=5e-4, mixedPrecision=True):
    
    torch.cuda.empty_cache()
    if mixedPrecision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    ###############################################################
    ###############################################################
    ## IMPORT DATA ## IMPORT DATA ### IMPORT DATA ## IMPORT DATA ##
    ###############################################################
    ###############################################################
    
    
    TrainX, ValX = train_test_split(InputTextList, random_state=2, test_size=0.1)
    
    TrainDS = tf.data.Dataset.from_tensor_slices((TrainX))
    ValDS = tf.data.Dataset.from_tensor_slices((ValX))

    ###############################################################
    ###############################################################
    ## SET UP MODEL # SET UP MODEL # SET UP MODEL # SET UP MODEL ##
    ###############################################################
    ###############################################################
    
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        baseModel,
        sequence_length=sequenceLength,
    )
    packer = preprocessor.packer
    tokenizer = preprocessor.tokenizer
    
    # keras.Layer to replace some input tokens with the "[MASK]" token
    masker = keras_nlp.layers.MaskedLMMaskGenerator(
        vocabulary_size=tokenizer.vocabulary_size(),
        mask_selection_rate=maskSelectionRate,
        mask_selection_length=maskSelectionLength,
        mask_token_id=tokenizer.token_to_id("[MASK]"),
        unselectable_token_ids=[
            tokenizer.token_to_id(x) for x in ["[CLS]", "[PAD]", "[SEP]"]
        ],
    )
    
    
    def preprocess(inputs):
        inputs = preprocessor(inputs)
        masked_inputs = masker(inputs["token_ids"])
        # Split the masking layer outputs into a (features, labels, and weights)
        # tuple that we can use with keras.Model.fit().
        features = {
            "token_ids": masked_inputs["token_ids"],
            "segment_ids": inputs["segment_ids"],
            "padding_mask": inputs["padding_mask"],
            "mask_positions": masked_inputs["mask_positions"],
        }
        labels = masked_inputs["mask_ids"]
        weights = masked_inputs["mask_weights"]
        return features, labels, weights
    
    # Use to create training and validation datasets. Run on a local machine, doing
    # this is pretty useless (but not harmful) as the runtime is roughly the same.
    # However, if the data either cannot fit in memory (which to be fair, it has
    # to given that I created the original data from tensor slices and not a file)
    # this improves performance by allowing data fetching, preprocessing, and training
    # to occur at the same time.
    PPTrainDS = TrainDS.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=batchSize,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    PPValDS = ValDS.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=batchSize,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    backbone = keras_nlp.models.BertBackbone.from_preset(baseModel)
    
    # Language modeling head
    mlm_head = keras_nlp.layers.MaskedLMHead(
        embedding_weights=backbone.token_embedding.embeddings,
    )
    
    inputs = {
        "token_ids": keras.Input(shape=(None,), dtype=tf.int32),
        "segment_ids": keras.Input(shape=(None,), dtype=tf.int32),
        "padding_mask": keras.Input(shape=(None,), dtype=tf.int32),
        "mask_positions": keras.Input(shape=(None,), dtype=tf.int32),
    }
    
    # Encoded token sequence
    sequence = backbone(inputs)
    
    outputs = mlm_head(sequence["sequence_output"], mask_positions=inputs["mask_positions"])
    
    # Define and compile our pretraining model.
    pretrainingModel = keras.Model(inputs, outputs)
    pretrainingModel.summary()
    pretrainingModel.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.adamw_experimental.AdamW(learning_rate=learningRate),
        weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
        jit_compile=True,
    )
    
    # Pretrain on our inputs
    pretrainingModel.fit(
        PPTrainDS,
        validation_data=PPValDS,
        epochs=trainingEpochs,
        shuffle=True
    )
    
    # This backbone can now be loaded into other applications to use as a domain
    # version of a language model.
    backbone.save(modelSavePath)


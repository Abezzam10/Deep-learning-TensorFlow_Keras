FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
        .learningRate(5e-5)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.NESTEROVS)
        .seed(seed)
        .build();
ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
        .fineTuneConfiguration(fineTuneConf)
        .setFeatureExtractor(featurizeExtractionLayer)
        .removeVertexKeepConnections("predictions")
        .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096).nOut(NUM_POSSIBLE_LABELS)//2 .weightInit(WeightInit.XAVIER) .activation(Activation.SOFTMAX).build(), featurizeExtractionLayer) .build();

public GraphBuilder setFeatureExtractor(String... layerName) {
    this.hasFrozen = true;
    this.frozenOutputAt = layerName;
    return this;
}


ZooModel zooModel = new VGG16();
ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
log.info(pretrainedNet.summary());

"""Test and train"""

DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);
int iEpoch = 0;
int i = 0;
while (iEpoch < EPOCH) {
    while (trainIterator.hasNext()) {
        DataSet trained = trainIterator.next();
        vgg16Transfer.fit(trained);
        if (i % SAVED_INTERVAL == 0 && i != 0) {
            ModelSerializer.writeModel(vgg16Transfer, new File(SAVING_PATH), false);
            evalOn(vgg16Transfer, devIterator, i);
        }
        i++;
    }
    trainIterator.reset();
    iEpoch++;
    evalOn(vgg16Transfer, testIterator, iEpoch);
}
package neronmine.craft;
import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.factory.Nd4j;
/**
 * Hello world!
 *
 */


public class App 
{
    public static void main( String[] args ) throws IOException
    {
        int numInput = 64; // количество входных нейронов
        int numHidden = 32; // количество скрытых нейронов
        int numOutput = 6; // количество выходных нейронов 
        int numEpochs = 10; // количество эпох обучения
        double learningRate = 0.001; // скорость обучения

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(numInput)
                .nOut(numHidden)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(numHidden)
                .nOut(numOutput)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // Логирование результатов каждые 10 итераций

        DataSetIterator mnistTrain = new MnistDataSetIterator(64, true, 12345);
        for (int i = 0; i < numEpochs; i++) {
            while (mnistTrain.hasNext()) {
                DataSet next = mnistTrain.next();
                model.fit(next);
            }
            mnistTrain.reset();
        }
        
        double[][] dataSides = {};
        



        INDArray inputData = Nd4j.create(dataSides);// Ваш входной массив данных
        INDArray output = model.output(inputData);
        int predictedClass = Nd4j.argMax(output, 1).getInt(0);
        System.out.println(predictedClass);
    }
}

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class RBFNN {
    Neuron[] neurons;
    OutputNeuron[] outputNeurons;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> testData = new ArrayList<int[]>();
    ArrayList<int[]> validationData = new ArrayList<int[]>();
    HashMap<Integer, Double> outputVector = new HashMap<Integer, Double>();
    HashMap<Integer, Double> activatedOutput = new HashMap<Integer, Double>();
    HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
    String dataSetName = "";
    
    double eta = .05;
    double performance = 0;
    
    public static void main(String[] args) throws IOException {
        String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
        for(String set: dataSets){
            System.out.println(set);
            for(int i = 1; i < 6; i++){
                RBFNN network = new RBFNN(set, i);
                network.trainRBFNN();
                network.testPerformance();
            }
        }
    }

    
    public RBFNN(String name, int index) throws IOException{
        this.dataSetName = name;
        fillTrainFile(index);
        fillTestFile(index);
        fillValidationSet();
        countClasses();
        neurons = new Neuron[classCounts.size()];
        outputNeurons = new OutputNeuron[classCounts.size()];
        int i = 0;
        for(Integer key : classCounts.keySet()){
            neurons[i] = new Neuron(key, trainingData);
            neurons[i].setCentroid();
            outputNeurons[i] = new OutputNeuron(classCounts.size(), key);
            i++;
        }
    }
    
    public double testPerformance(){
        double performance = 0;
        for(int[] arr : testData){
            for(Neuron n: neurons){
                n.RBF(arr);
            }
            double max = Double.MIN_VALUE;
            int classVal = 0;
            for(OutputNeuron on : outputNeurons){
                int j = 0;
                while(j < on.weights.length){
                    on.output += on.weights[j]*neurons[j].basisOutput;
                    j++;
                }
                on.logistic();
                if(on.output > max){
                    max = on.output;
                    classVal = on.classVal;
                }
            }
            if(classVal == arr[arr.length-1]){
                performance++;
            }
        }
        System.out.println(performance/((double)testData.size()));
        return performance/((double)testData.size());
    }
    
    public void trainRBFNN(){
        int k = 0;
        while(validatePerformance() < .95 && k < 1000){
            for(OutputNeuron on : outputNeurons){
                on.weightsCopy = on.weights;
                for(int[] arr : trainingData){
                    for(Neuron n: neurons){
                        n.RBF(arr);
                    }
                    int j = 0;
                    while(j < on.weights.length){
                        on.output += on.weights[j]*neurons[j].basisOutput;
                        j++;
                    }
                    on.logistic();
                    if(arr[arr.length-1] == on.classVal){
                        for(int i = 0; i < neurons.length; i++){
                            double err = 0;
                            err = (1-on.output)*on.output*(1-on.output)*neurons[i].basisOutput;
                            on.weights[i] += eta*err;
                        }
                    }else{
                        for(int i = 0; i < neurons.length; i++){
                            double err = 0;
                            err = (0-on.output)*on.output*(1-on.output)*neurons[i].basisOutput;
                            on.weights[i] += eta*err;
                        }
                    }
                }
            }
            k++;
        }
    }
    
    public double validatePerformance(){
        double performance = 0;
        for(int[] arr : validationData){
            for(Neuron n: neurons){
                n.RBF(arr);
            }
            double max = Double.MIN_VALUE;
            int classVal = 0;
            for(OutputNeuron on : outputNeurons){
                int j = 0;
                while(j < on.weights.length){
                    on.output += on.weights[j]*neurons[j].basisOutput;
                    j++;
                }
                on.logistic();
                if(on.output > max){
                    max = on.output;
                    classVal = on.classVal;
                }
            }
            if(classVal == arr[arr.length-1]){
                performance++;
            }
        }
        return performance/((double)validationData.size());
    }
    
    
    /************************************************************
    Counts all of the classes in the training set. 
    ************************************************************/
    
    public void countClasses(){
        for(int i = 0; i < trainingData.size(); i++){
            if(classCounts.containsKey(trainingData.get(i)[trainingData.get(i).length-1])){ // hashmap contains class?
                int currentVal = classCounts.get(trainingData.get(i)[trainingData.get(i).length-1]); // get current count
                currentVal++; // increment count by 1
                classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], currentVal); //update map.
            }else{
                classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], 1); // add class to map
            }
        }
    }
    
    /************************************************************
    Fills the training file based on the index that is to be skipped.
    Used specifically for cross validation.
    ************************************************************/
    
    void fillTrainFile(int indexToSkip) throws IOException{
        for(int i = 1; i < 6; i++){
            if(i == indexToSkip) continue; // skip file at index.
            Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
            while(fileScanner.hasNextLine()){
                String[] arr = fileScanner.nextLine().split(" ");

                int[] vals = new int[arr.length];
                for(int j = 0; j < arr.length; j++){
                    vals[j] = Integer.parseInt(arr[j]);
                }
                trainingData.add(vals);
            }
            fileScanner.close();
        }
    }
    
    /************************************************************
    Fills test file similarly to fillTrainFile.
    ************************************************************/
    
    void fillTestFile(int indexToSkip) throws IOException{
        Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+indexToSkip+".txt"));
        while(fileScanner.hasNextLine()){
            String[] arr = fileScanner.nextLine().split(" ");
            int[] vals = new int[arr.length];
            for(int j = 0; j < arr.length; j++){
                vals[j] = Integer.parseInt(arr[j]);
            }
            testData.add(vals);
        }
        fileScanner.close();
    }
    
    /************************************************************
    Fills the validation set.
    ************************************************************/
    void fillValidationSet() throws IOException{
        Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/validationSet.txt"));
        while(fileScanner.hasNextLine()){
            String[] arr = fileScanner.nextLine().split(" ");

            int[] vals = new int[arr.length];
            for(int j = 0; j < arr.length; j++){
                vals[j] = Integer.parseInt(arr[j]);
            }
            validationData.add(vals);
        }
        fileScanner.close();
    }
    
    public void printClassCounts(){
        for(Integer key : classCounts.keySet()){
            System.out.println("Class : " + key);
        }
    }
    
    class Cluster {
        ArrayList<int[]> values = new ArrayList<int[]>();
        double[] centroid;
        
        void setCentroid(){
            for(int[] arr : values){
                for(int i = 0; i<arr.length; i++){
                    centroid[i] += arr[i];
                }
            }
            for(int i = 0; i < centroid.length; i++){
                centroid[i] /= (double)values.size();
            }
        }
    }
}

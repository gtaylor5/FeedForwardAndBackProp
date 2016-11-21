import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class RBFNN {
    Neuron[] neurons;
    Cluster[] clusters;
    OutputNeuron[] outputNeurons;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> trainingDataCopy = new ArrayList<int[]>();
    ArrayList<int[]> testData = new ArrayList<int[]>();
    ArrayList<int[]> validationData = new ArrayList<int[]>();
    HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
    String dataSetName = "";
    int k = 0;
    
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

    /************************************************************
    Constructor for RBFNN. 
    ************************************************************/
    
    public RBFNN(String name, int index) throws IOException{
        //Set k values for each data set. Used guess and check.
        if(name.equalsIgnoreCase("soybean")){
            this.k = 6;
        }else if(name.equalsIgnoreCase("Iris")){
            this.k = 8;
        }else if(name.equalsIgnoreCase("glassid")){
            this.k = 75;
        }else if(name.equalsIgnoreCase("breastcancer")){
            this.k = 2;
        }else{
            this.k = 6;
        }

        this.dataSetName = name;
        fillTrainFile(index); // fills training file.
        fillTestFile(index); //fills test file
        fillValidationSet(); //fills validation set
        trainingDataCopy = trainingData;
        countClasses(); //counts classes in the dataset
        initializeNeurons();//initializes the neurons
        
        outputNeurons = new OutputNeuron[classCounts.size()]; //set number of output nodes equal to number of classes
        int i = 0;
        for(Integer key: classCounts.keySet()){ //initialize each output node with size of weight vector and an associated class.
            outputNeurons[i] = new OutputNeuron(neurons.length, key);
            i++;
        }
    }
    
    public RBFNN(String name) {
        this.dataSetName = name;
      //Set k values for each data set. Used guess and check.
        if(name.equalsIgnoreCase("soybean")){
            this.k = 6;
        }else if(name.equalsIgnoreCase("Iris")){
            this.k = 8;
        }else if(name.equalsIgnoreCase("glassid")){
            this.k = 75;
        }else if(name.equalsIgnoreCase("breastcancer")){
            this.k = 2;
        }else{
            this.k = 6;
        }
        
        

    }
    
    void initialize(){
        outputNeurons = new OutputNeuron[classCounts.size()]; //set number of output nodes equal to number of classes
        int i = 0;
        for(Integer key: classCounts.keySet()){ //initialize each output node with size of weight vector and an associated class.
            outputNeurons[i] = new OutputNeuron(neurons.length, key);
            i++;
        }
    }

    /************************************************************
    Initialize the number of clusters and neurons. Sets the neurons
    center to be a cluster centroid.
    ************************************************************/
    
    public void initializeNeurons(){
        
        clusters = new Cluster[k];
        initializeClusters(); // initialize clusters method below
        fillClusters(); //see fill clusters method below
        neurons = new Neuron[k];
        for(int i = 0; i < k; i++){
            neurons[i] = new Neuron(trainingData); // pass in training data
            neurons[i].center = clusters[i].centroid; //set centroid.
        }
    }
    
    /************************************************************
    Initializes clusters with random data point from training set
    ************************************************************/
    
    public void initializeClusters(){
        for(int i = 0; i < clusters.length; i++){
            int random = (int)Math.random()*trainingDataCopy.size();
            clusters[i] = new Cluster(trainingDataCopy.get(random));
            trainingDataCopy.remove(random);
        }
    }
    
    /************************************************************
    Assigns the training data to the correct cluster. Used for
    calculating the centroids of the cluster.
    ************************************************************/
    
    public void fillClusters(){
        for(int[] arr : trainingData){
            double min = Double.MAX_VALUE;
            int index = 0;
            for(int i = 0; i < clusters.length; i++){
                clusters[i].euclidDistance(arr);
                if(clusters[i].distance < min){
                    min = clusters[i].distance;
                    index = i;
                }
            }
            clusters[index].values.add(arr);
            clusters[index].setCentroid();
        }
    }
    
    /************************************************************
    Tests overall performance of the learned models with the test
    set. Identical to validation method below.
    ************************************************************/
    
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
        System.out.printf("RBFNN: %.2f",100*performance/((double)testData.size()));
        System.out.println();
        return performance/((double)testData.size());
    }
    
    /************************************************************
    Method used to train the radial basis function neural network
    ************************************************************/
    
    public void trainRBFNN(){
        int k = 0;
        //running conditions based on performance or number of iterations.
        while(validatePerformance() < .95 && k < 500){
            //train each output neuron separately.
            for(OutputNeuron on : outputNeurons){
                on.weightsCopy = on.weights;
                for(int[] arr : trainingData){
                    //calculate basis output.
                    for(Neuron n: neurons){
                        n.RBF(arr);
                    }
                    int j = 0;
                    //find the net output.
                    while(j < on.weights.length){
                        on.output += on.weights[j]*neurons[j].basisOutput;
                        j++;
                    }
                    //activate
                    on.logistic();
                    
                    //positive case (target = 1)
                    if(arr[arr.length-1] == on.classVal){
                        for(int i = 0; i < neurons.length; i++){
                            double err = 0;
                            err = (1-on.output)*on.output*(1-on.output)*neurons[i].basisOutput; //calculate gradient error
                            on.weights[i] += eta*err; //update weights.
                        }
                    }else{ // negative case (target = 0)
                        for(int i = 0; i < neurons.length; i++){
                            double err = 0;
                            err = (0-on.output)*on.output*(1-on.output)*neurons[i].basisOutput; // calculate gradient error
                            on.weights[i] += eta*err; //update weights.
                        }
                    }
                }
            }
            k++;
        }
    }
    
    /************************************************************
    Test the performance of the learned weights against a 
    validation set. Returns the performance as a decimal. Used
    primarily in training.
    ************************************************************/
    
    public double validatePerformance(){
        double performance = 0;
        for(int[] arr : validationData){
            //calculate RBF output for each neuron.
            for(Neuron n: neurons){
                n.RBF(arr);
            }
            double max = Double.MIN_VALUE;
            int classVal = 0;
            for(OutputNeuron on : outputNeurons){
                int j = 0;
                //calculate the output for each output node.
                while(j < on.weights.length){
                    on.output += on.weights[j]*neurons[j].basisOutput;
                    j++;
                }
                //pass it through an activation function.
                on.logistic();
                //find the class with the maximum value.
                if(on.output > max){
                    max = on.output;
                    classVal = on.classVal;
                }
            }
            
            //if the classes match, increment performance counter.
            if(classVal == arr[arr.length-1]){
                performance++;
            }
        }
        //return performance as a decimal.
        return performance/((double)validationData.size());
    }
    
    /************************************************************
    
    HELPER METHODS
    
    
    ************************************************************/
    
    
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
    
    
    void printWeights(){
        for(OutputNeuron c : outputNeurons){
            Main.writer2.print("Weights for Class " + c.classVal + " : ");
            for(double val : c.weights){
                Main.writer2.print(val + " ");
            }
            Main.writer2.println();
        }
    }
    
    
    /************************************************************
    Helper Class for K-Means.
    ************************************************************/
    
    
    class Cluster {
        ArrayList<int[]> values = new ArrayList<int[]>();
        double[] centroid;
        double distance = 0;
        
        /************************************************************
        Constructor
        ************************************************************/
        
        public Cluster(int[] arr){
            values.add(arr);
            this.centroid = new double[arr.length-1];
            setCentroid();
        }
        
        /************************************************************
        Recalculates the centroid of the cluster.
        ************************************************************/
        
        void setCentroid(){
            for(int i = 0; i < centroid.length; i++){
                centroid[i] = 0;
            }
            for(int[] arr : values){
                for(int i = 0; i<arr.length-1; i++){
                    centroid[i] += arr[i];
                }
            }
            for(int i = 0; i < centroid.length; i++){
                centroid[i] /= (double)values.size();
            }
        }
        
        /************************************************************
        Calculate Euclidean Distance between a query point and the
        centroid.
        ************************************************************/
        
        void euclidDistance(int[] queryPoint){
            distance = 0;
            for(int i = 0; i < queryPoint.length-1; i++){
                distance += Math.pow((((double)queryPoint[i])-centroid[i]),2);
            }
        }
    }
}

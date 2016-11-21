import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;


/************************************************************
Feed Forward Neural Network with Backpropagation as the 
training method.
************************************************************/

public class FeedForwardNN {
    
    OutputNode[] outputNodes; // output nodes
    HiddenNode[] hiddenNodes; // hidden nodes

    ArrayList<int[]> trainingData = new ArrayList<int[]>(); // training data as array list
    ArrayList<int[]> trainingDataCopy = new ArrayList<int[]>(); //copy of training data
    ArrayList<int[]> testData = new ArrayList<int[]>(); //test data
    ArrayList<int[]> validationData = new ArrayList<int[]>(); //validation set
    HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>(); //class occurrences.
    
    String dataSetName = ""; // data set to be used.
    Double totalError = 0.0; //error of the whole network.
    
    public static void main(String[] args) throws IOException {
        String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
       //for(String set: dataSets){
            //System.out.println(set);
            for(int i = 3; i < 4; i++){
                FeedForwardNN network = new FeedForwardNN("BreastCancer", i);
                network.trainNetwork();
                network.test();
            }
       // }
    }
    
    /**************************************************************************
     * Constructor
    **************************************************************************/
    
    public FeedForwardNN(String name, int index) throws IOException{
        this.dataSetName = name; // data set name
        fillTestFile(index);
        fillTrainFile(index);
        fillValidationSet();
        countClasses(); //count classes in data set.
        outputNodes = new OutputNode[classCounts.size()]; //initialize number of output nodes to equal number of classes
        hiddenNodes = new HiddenNode[classCounts.size()]; // initialize hidden neurons equal to number of classes
        
        int i = 0; 
        //assume number of hiddent nodes is equal to the number of classes.
        for(Integer key : classCounts.keySet()){
            hiddenNodes[i] = new HiddenNode(trainingData.get(0).length-1); // construct hidden node
            outputNodes[i] = new OutputNode(hiddenNodes.length, key); // construct output node
            i++;
        }
        
    }
    
    public FeedForwardNN(String dataSetName2) {
        this.dataSetName = dataSetName2;
        
    }
    
    void initialize(){
        countClasses();
        outputNodes = new OutputNode[classCounts.size()]; //initialize number of output nodes to equal number of classes
        hiddenNodes = new HiddenNode[classCounts.size()]; // initialize hidden neurons equal to number of classes
        
        int i = 0; 
        //assume number of hiddent nodes is equal to the number of classes.
        for(Integer key : classCounts.keySet()){
            hiddenNodes[i] = new HiddenNode(trainingData.get(0).length-1); // construct hidden node
            outputNodes[i] = new OutputNode(hiddenNodes.length, key); // construct output node
            i++;
        }
    }

    /**************************************************************************
     * Trains Feed Forward Network using Backpropagation.
    **************************************************************************/
    
    public void trainNetwork(){
        double currentPerformance = validationPerformance(); //initialize performance
        double maxPerformance = 0; //set max performance 
        OutputNode[] onCopy = outputNodes;
        HiddenNode[] hnCopy = hiddenNodes;
        int q = 0; // iterator variable.
        while(currentPerformance >= maxPerformance && q < 5){ // main loop
            onCopy = outputNodes;
            hnCopy = hiddenNodes;
            maxPerformance = currentPerformance; // set max performance equal to the current performance
            int it = 0; 
            while(it < 10000){ // number of times to train against the data set
                for(int f = 0; f < trainingData.size(); f++){ //iterate through training set
                    int[] arr = trainingData.get(f);
                        totalError = 0.0;
                        
                        //calculate outputs for each hidden node.
                        
                        for(HiddenNode h : hiddenNodes){
                            h.calculateNet(arr);
                            h.calculateOutput();
                        }
                        
                        //store outputs in array for later use.
                        
                        double[] hiddenNodeOutputs = new double[hiddenNodes.length];
                        for(int i = 0; i < hiddenNodes.length; i++){
                            hiddenNodeOutputs[i] = hiddenNodes[i].output;
                        }
                        
                        //Calculate outputs for output nodes and update weights.
                        
                        for(OutputNode on : outputNodes){
                            on.calculateNet(hiddenNodeOutputs);
                            on.calculateOutput();
                            if(on.classVal == arr[arr.length-1]){
                                on.calculateError(1.0);
                                on.updateWeights(hiddenNodeOutputs, 1.0);
                                
                            }else{
                                on.calculateError(0);
                                on.updateWeights(hiddenNodeOutputs, 0);
                            }
                            totalError += on.error;
                        }
                        
                        //back propagate to update the weights of the hidden nodes.
                        
                        for(int i = 0; i < hiddenNodes.length; i++){
                            double Etot = 0;
                            for(int j = 0; j < outputNodes.length; j++){
                                Etot += outputNodes[j].dEdO*outputNodes[j].dOdN*outputNodes[j].copyWeights[i]; //calculate total error.
                            }
                            for(int j = 0; j < arr.length-1; j++){
                                hiddenNodes[i].setPartials(((double)arr[j])); // calculate partial derivatives based on feature
                                hiddenNodes[i].dEdW = Etot*hiddenNodes[i].dNdW*hiddenNodes[i].dOdN; //set gradient of the error with respect to the weight.
                                hiddenNodes[i].weights[j] -= hiddenNodes[i].eta*hiddenNodes[i].dEdW; // update corresponding weight.
                            }
                        }
                    }
            it++;
            }
            currentPerformance = validationPerformance();
            if(currentPerformance > .8){ // break early if performance is greater than 80%
                return;
            }
            q++;
        }
        outputNodes = onCopy;
        hiddenNodes = hnCopy;
    }
    
    /**************************************************************************
     * Tests the learned network against the test set.
    **************************************************************************/
    
    public double test() {
        double performance = 0;
        for(int[] arr: testData){
            
            //calculate hidden node outputs
            
            for(HiddenNode h : hiddenNodes){
                h.calculateNet(arr);
                h.calculateOutput();
            }
            
            //store outputs in array.
            
            double[] hiddenNodeOutputs = new double[hiddenNodes.length];
            for(int i = 0; i < hiddenNodes.length; i++){
                hiddenNodeOutputs[i] = hiddenNodes[i].output;
            }
            
            //find the class with the highest value output.
            
            int classVal = 0;
            double max = Double.MIN_VALUE;
            for(OutputNode on : outputNodes){
                on.calculateNet(hiddenNodeOutputs);
                on.calculateOutput();
                if(on.output > max){
                    max = on.output;
                    classVal = on.classVal;
                }
            }
            
            //if the classes match, incremement performance.
            
            if(classVal == arr[arr.length-1]){
                performance++;
            }
        }
        
        //calculate performance.
        
        System.out.printf("Feed Forward: %.2f",100*performance/((double)testData.size()));
        System.out.println();
        return performance/((double)testData.size());
    }
    
    /**************************************************************************
     * Tests against validation set. Same as test() above but with validation
     *  set
    **************************************************************************/

    public double validationPerformance() {
        double performance = 0;
        for(int[] arr: validationData){
            for(HiddenNode h : hiddenNodes){
                h.calculateNet(arr);
                h.calculateOutput();
            }
            double[] hiddenNodeOutputs = new double[hiddenNodes.length];
            for(int i = 0; i < hiddenNodes.length; i++){
                hiddenNodeOutputs[i] = hiddenNodes[i].output;
            }
            int classVal = 0;
            double max = Double.MIN_VALUE;
            for(OutputNode on : outputNodes){
                on.calculateNet(hiddenNodeOutputs);
                on.calculateOutput();
                if(on.output > max){
                    max = on.output;
                    classVal = on.classVal;
                }
            }
            if(classVal == arr[arr.length-1]){
                performance++;
            }
        }
        //System.out.println(performance/((double)validationData.size()));
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
        Main.writer2.println();
        for(OutputNode c : outputNodes){
            Main.writer2.print("Weights for Class " + c.classVal + " : ");
            for(double val : c.weights){
                Main.writer2.print(val + " ");
            }
            Main.writer2.println();
        }
        Main.writer2.println();
        for(HiddenNode n : hiddenNodes){
            Main.writer2.print("Weights for Hidden Node : ");
            for(double val : n.weights){
                Main.writer2.print(val + " ");
            }
            Main.writer2.println();
        }
        Main.writer2.println();
    }
    
}

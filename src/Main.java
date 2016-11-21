import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

class Main{

    ArrayList<int[]> trainingData = new ArrayList<int[]>(); //training data as usable arraylist
    ArrayList<int[]> testData = new ArrayList<int[]>(); //test data as usable arraylist
    ArrayList<int[]> validationSet = new ArrayList<int[]>(); //validation set as usable arraylist
    String dataSetName; //the name of the dataset being tested/trained
      
    ArrayList<Double> FFNNPerformance = new ArrayList<Double>();
    ArrayList<Double> RBFNNPerformance = new ArrayList<Double>();
  
    public Main(String name){
      this.dataSetName = name;
    }
    
    static PrintWriter writer2;
  
    public static void main(String[] args) throws IOException{
          String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
          writer2 = new PrintWriter(new FileWriter("LearnedModels.txt", true));
              for(int i = 0; i < dataSets.length; i++){ //Change this line to see the other learned models for the other data sets.
                  System.out.println(dataSets[i]);
                  writer2.println(dataSets[i]);
                  writer2.println();
                  Main m = new Main(dataSets[i]);
                  for(int j = 1; j < 6; j++){
                      writer2.println("Fold #: "+j);
                      writer2.println();
                      m.fillTestFile(j);
                      m.fillTrainFile(j);
                      m.fillValidationSet();
                      m.test();
                      m.reset();
                      writer2.println();
                  }
                  System.out.println();
                m.writeStatistics();
                writer2.println("-------------------");
          }
       writer2.close();
    }
    
    public void writeStatistics() throws IOException{
        PrintWriter writer = new PrintWriter(new FileWriter("StatisticalResults.txt",true));
        
        HashMap<String,ArrayList<Double>> arr = new HashMap<String,ArrayList<Double>>();
        arr.put("Feed Forward w/ Backpropagation",FFNNPerformance);
        arr.put("Radial Basis Function NN",RBFNNPerformance);
        
        writer.println("Data set: " + this.dataSetName);
        writer.println();
        for(String key : arr.keySet()){
            writer.println("Statistics for: " + key);
            writer.println("Mean Performance: " + mean(arr.get(key)));
            writer.println("Variance of Performance: " + variance(arr.get(key)));
            writer.println("Standard Deviation: "+ Math.sqrt(variance(arr.get(key))));
            double standardError = Math.sqrt(variance(arr.get(key)))/Math.sqrt(arr.get(key).size());
            double margin = standardError/2.0;
            writer.println("Confidence Interval: " + (mean(arr.get(key))-margin) + " to " + (mean(arr.get(key))+margin));
            writer.println();
        }
        writer.println("----------------------------------------");
        writer.close();
    }
    
    double variance(ArrayList<Double> arr){
        double sum = 0;
        for(Double val: arr){
            sum+=Math.pow(val-mean(arr),2);
        }
        return sum/((double) arr.size());
    }
    
    double mean(ArrayList<Double> arr){
        double sum = 0;
        for(Double val : arr){
            sum+=val;
        }
        return sum/((double)arr.size());
    }
  
    void reset(){
        this.testData.removeAll(testData);
        this.trainingData.removeAll(trainingData);
        this.validationSet.removeAll(validationSet);
    }
  
    void test() throws IOException{
    
      //Feed Forward NN w/ Backprop
        FeedForwardNN net = new FeedForwardNN(this.dataSetName);
        net.trainingData = this.trainingData;
        net.testData = this.testData;
        net.validationData = this.validationSet;
        net.initialize();
        writer2.println("Feed Forward NN ");
        net.trainNetwork();
        net.printWeights();
        FFNNPerformance.add(net.test());
        
      //Radial Basis Function
        RBFNN rbf = new RBFNN(this.dataSetName);
        rbf.trainingData = this.trainingData;
        rbf.testData = this.testData;
        rbf.validationData = this.validationSet;
        rbf.trainingDataCopy = rbf.trainingData;
        rbf.countClasses();
        rbf.initializeNeurons();
        rbf.initialize();
        rbf.trainRBFNN();
        writer2.println("RBFNN ");
        rbf.printWeights();
        RBFNNPerformance.add(rbf.testPerformance());
    
     
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
            validationSet.add(vals);
        }
        fileScanner.close();
    }

}

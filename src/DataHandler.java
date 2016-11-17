import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class DataHandler {

    String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
    String[] unProcessedFiles = {"soybeanunprocessed","irisunprocessed", "glassunprocessed", 
            "breast-cancer-wisconsinunprocessed","house-votes-84unprocessed"};
    String[] processedFiles = {"soybeanprocessed.txt","irisprocessed.txt", "glassprocessed.txt", 
            "breastcancerprocessed.txt","house-votes-84processed.txt"};
    ArrayList<PreProcessTask> tasks = new ArrayList<PreProcessTask>();
    ArrayList<String[]> fileAsArray = new ArrayList<String[]>();
    HashMap<String, Integer> classCounts = new HashMap<String, Integer>();
    
    /************************************************************
    
    processData() takes the unprocessed files from the unProcessedFiles
    array and processes them into binary arrays for each data entry and
    stores them in the corresponding processedFiles array element.
     * @throws IOException 
    
    ************************************************************/
    public static void main(String[] args) throws IOException {
        DataHandler h = new DataHandler();
        h.processData();
    }
    
    /************************************************************
    Method to process all of the data.
    ************************************************************/
    
    public void processData() throws IOException{
        for(int i = 0 ; i < dataSets.length; i++){
            PreProcessTask task = new PreProcessTask(dataSets[i]);
            String path = "Data/"+dataSets[i]+"/"+dataSets[i]+"Processed.txt";
            File file = new File(path);
            file.getParentFile().mkdirs();
            file.createNewFile();
            PrintWriter writer = new PrintWriter(file);
            try {
                task.storeFileInArray("Unprocessed/"+unProcessedFiles[i]+".txt");
                task.booleanizeData();
                for(int j = 0; j < task.booleanizedFile.size();j++){
                    for(int k = 0; k < task.booleanizedFile.get(j).length; k++){
                        writer.print(task.booleanizedFile.get(j)[k] + " ");
                    }
                    writer.println();
                }
                convertToString(task.booleanizedFile);
                countClasses();
                createValidationSet(dataSets[i]);
                splitData(dataSets[i]);
            }catch (FileNotFoundException e) {
                e.printStackTrace();
            }finally{
                writer.close();
            }
        }
    }
    
    /**************************************************************
     * Parse file and store in arraylist as string array.
     *************************************************************/
    public void storeFileAsArray(String filePath) throws FileNotFoundException{
        File file = new File(filePath);
        System.out.println(filePath);
        Scanner fileScanner = new Scanner(file);
        int j = 0;
        while(fileScanner.hasNextLine()){
            j++;
            String[] line = fileScanner.nextLine().split(" ");
            System.out.println(line.length);
            fileAsArray.add(line);
        }
        System.out.println(j);
        fileScanner.close();
    }
    
    /************************************************************
    Splits data into 5 equal portions for cross validation.
    ************************************************************/
    
    public void splitData(String dataSetName) throws IOException{
        int originalSize = fileAsArray.size();
        for(int i = 0; i < 5; i++){ // number of files
            ArrayList<String[]> temp = new ArrayList<String[]>(); // file as Array
            if(i == 4){
                while(fileAsArray.size()!=0){
                    temp.add(fileAsArray.get(0));
                    fileAsArray.remove(0);
                }
                writeToFile(i,temp, dataSetName);
            }else{
                while(temp.size() != originalSize/5){
                    int randomIndex = (int)(Math.random()*fileAsArray.size());
                    temp.add(fileAsArray.get(randomIndex));
                    fileAsArray.remove(randomIndex);
                }
                writeToFile(i,temp, dataSetName);
            }
        }
    }
    
    /**************************************************************
     * Create Validation Set
     * @throws IOException 
     *************************************************************/
    public void createValidationSet(String dataSetName) throws IOException{
        ArrayList<String[]> validationSet = new ArrayList<String[]>();
        for(int i = 0; i < fileAsArray.size()*.1; i++){
            int randomIndex = (int)(Math.random()*fileAsArray.size());
            validationSet.add(fileAsArray.get(randomIndex));
            fileAsArray.remove(randomIndex);
        }
        classCounts.clear();
        countClasses();
        writeToFile(validationSet, dataSetName);
    }
    
    /************************************************************
    Count occurrences of each class.
    ************************************************************/
    
    public void countClasses(){
        //handle special cases.
        for(int i = 0; i < fileAsArray.size(); i++){
            if(classCounts.containsKey(fileAsArray.get(i)[fileAsArray.get(i).length-1])){ // hashmap contains class?
                int currentVal = classCounts.get(fileAsArray.get(i)[fileAsArray.get(i).length-1]); // get current count
                currentVal++; // increment count by 1
                classCounts.put(fileAsArray.get(i)[fileAsArray.get(i).length-1], currentVal); //update map.
            }else{
                classCounts.put(fileAsArray.get(i)[fileAsArray.get(i).length-1], 1); // add class to map
            }
        }
        //printClassCounts(); // prints class counts.
    }
    
    /************************************************************
    Writes to file. Used to create 5 different equally sized sets.
    ************************************************************/
    
    public void writeToFile(int fileNum, ArrayList<String[]> array, String dataSetName) throws IOException{
        String path = "Data/"+dataSetName+"/Set"+(fileNum+1)+".txt";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        PrintWriter writer = new PrintWriter(new FileWriter(file, true));
        for(int i = 0; i < array.size(); i++){
            for(int j = 0; j < array.get(i).length; j++){
                writer.print(array.get(i)[j]+ " ");
            }
            writer.println();
        }
        writer.close();
    }
    
    /************************************************************
    Writes to validation set.
    ************************************************************/
    
    public void writeToFile(ArrayList<String[]> array, String dataSetName) throws IOException{
        String path = "Data/"+dataSetName+"/validationSet.txt";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        PrintWriter writer = new PrintWriter(new FileWriter(file, true));
        for(int i = 0; i < array.size(); i++){
            for(int j = 0; j < array.get(i).length; j++){
                writer.print(array.get(i)[j]+ " ");
            }
            writer.println();
        }
        writer.close();
    }
    
    /************************************************************
    Converts array list of integer arrays to a string arrays.
    ************************************************************/
    
    void convertToString(ArrayList<int[]> list){
        for(int[] arr : list){
            String[] str = new String[arr.length];
            for(int i = 0; i < arr.length; i++){
                str[i] = Integer.toString(arr[i]);
            }
            fileAsArray.add(str);
        }
    }
    
}

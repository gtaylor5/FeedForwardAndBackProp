import java.util.ArrayList;
import java.util.HashMap;

/**************************************************************************
 * 
 * This class serves as a framework for the Neurons (hidden layer) of the
 * RBF Neural Network.
 * 
 * 
**************************************************************************/

public class Neuron {
    

    HashMap<Integer, Double> weights = new HashMap<Integer, Double>();
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    Integer classVal;
    double[] center;
    double sigma = 1.5;
    double basisOutput;
    
    public Neuron(Integer classVal, ArrayList<int[]> train){
        this.classVal = classVal;
        this.trainingData = train;
        this.center = new double[train.get(0).length-1];
    }
    
    /**************************************************************************
     * Calculate basis function given a query point.
    **************************************************************************/

    public void RBF(int[] queryPoint){
        double dotProd = euclidNorm(queryPoint);
        double sigmaSquared = sigma*sigma;
        double fractionTerm = (-1.0)/(2.0*sigmaSquared);
        double dotProdXFractionTerm = fractionTerm*dotProd;
        double exponentiate = Math.exp(dotProdXFractionTerm);
        basisOutput = exponentiate;
    }
 
    
    /**************************************************************************
     * Set cluster for this Neuron.
    **************************************************************************/
    
    public void setCentroid(){
        int classCount = 0;
        for(int[] arr : trainingData){
            if(arr[arr.length-1] == classVal){
                classCount++;
                for(int i = 0; i < center.length; i++){
                    center[i] += arr[i];
                }
            }
        }
        for(int i = 0; i < center.length; i++){
            center[i] /= classCount;
        }
    }
    
    /**************************************************************************
     * 
     * 
     * Helper Methods.
     * 
     * 
    **************************************************************************/
    
    public Double euclidNorm(int[] x){
        Double dist = 0.0;
        for(int i = 0; i < x.length-1; i++){
            dist += Math.pow((((double)x[i])-center[i]),2);
        }
        return dist;
    }
    
    public Double dot(int[] queryPoint){
        double sum = 0;
        for(int i = 0; i < queryPoint.length; i++){
            sum+= (((double)queryPoint[i])*center[i]);
        }
        return sum;
    }
    
    public void printCenter(){
        for(Double val : center){
            System.out.print(val +  " ");
        }
        System.out.println();
    }
}


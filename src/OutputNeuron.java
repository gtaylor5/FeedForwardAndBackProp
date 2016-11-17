
public class OutputNeuron {
    
    double[] weights;
    double[] weightsCopy;
    Integer classVal;
    Double output;
    
    public OutputNeuron(int numClasses, Integer classVal){
        this.classVal = classVal;
        weights = new double[numClasses];
        output = 0.0;
    }
    
    public void logistic(){
        output = 1.0/(1.0 + Math.exp(-output));
    }
    
}


/************************************************************
Output Node Class for the Feed forward neural network with
backpropagation.
************************************************************/

public class OutputNode {
    
    double[] weights; // weights.
    double[] copyWeights; //copy of the weights.
    double output = 0; //activated output
    double net = 0; // net output
    double error = 0; //error
    double eta = 0.75; //learning rate.
    
    double dEdO = 0; //derivative of error with respect to the output.
    double dOdN = 0; //derivative of the output with respect to the net output.
    
    int classVal; //classification of the output node.
    
    /************************************************************
    Constructor.
    ************************************************************/
    
    public OutputNode(int length, int classVal){
        this.weights = new double[length];
        this.classVal = classVal;
        
        for(int i = 0; i < weights.length; i++){
            weights[i] = -.001+Math.random()*.001; //initialize weights.
        }
        copyWeights = weights;
    }
    
    /************************************************************
    Calculates net output from query point. The query comes from
    a double array of the hidden node outputs.
    ************************************************************/
    
    public void calculateNet(double[] queryPoint){
        double val = 0;
        for(int i = 0; i < queryPoint.length-1; i++){
            val += weights[i]*queryPoint[i];
        }
        net = val;
    }
    
    /************************************************************
    Calculate activated output.
    ************************************************************/
    
    public void calculateOutput(){
        output = 1.0/(1.0+Math.exp(-net));
    }
    
    /************************************************************
    Calculate error.
    ************************************************************/
    
    public void calculateError(double target){
        this.error = .5*Math.pow((target-output),2);
    }
    
    /************************************************************
    Update weights based on the hidden output.
    ************************************************************/
    
    public void updateWeights(double[] hiddenOutputs, double target){
        copyWeights = weights;
        dEdO = (output-target);
        dOdN = output*(1-output);
        
        double[] updateValue = new double[hiddenOutputs.length];
        for(int i = 0; i < updateValue.length; i++){
            updateValue[i] = dEdO*dOdN*hiddenOutputs[i];
        }
        for(int i = 0; i < weights.length; i++){
            weights[i] -= eta*updateValue[i];
        }
    }
    
    public void printWeights(){
        for(double val : weights){
            System.out.print(val + " ");
        }
        System.out.println();
    }
    
}



/************************************************************
Hidden node class used in Feed Forward Neural Network.
************************************************************/

public class HiddenNode {
    
    double[] weights; // weights.
    double output = 0; // activated output
    double net = 0; // net output
    double dOdN = 0; //gradient of the output with respect to the net output.
    double dNdW = 0; //gradient of the net output with respect to the weight
    double dEdW = 0; //gradient of the Error with respect to the weight.
    double eta = .0002; // learning rate.
    
    /************************************************************
    Constructor
    ************************************************************/
    
    public HiddenNode(int length){
        this.weights = new double[length];
        
        for(int i = 0; i < weights.length; i++){
            weights[i] = -.001+Math.random()*.001; // initialize weights to small values.
        }
    }
    
    /************************************************************
    calculate net output based on input query point
    ************************************************************/
    
    public void calculateNet(int[] queryPoint){
        double val = 0;
        for(int i = 0; i < queryPoint.length-1; i++){
            val += weights[i]*((double)queryPoint[i]);
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
    set partial derivatives.
    ************************************************************/
    
    public void setPartials(double i){
        dOdN = output*(1-output);
        dNdW = i;
    }
    
    

}

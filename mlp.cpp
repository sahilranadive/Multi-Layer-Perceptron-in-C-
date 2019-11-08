#include <bits/stdc++.h>

using namespace std;


//declare global variables here
float alpha_for_relu = 0.05;
int input_dim;
vector<int> hidden_dim;
int output_dim;
int hidden_layers;
vector<int> activations;
//start defining activation functions here

float relu(float x){//Rectified Linear Unit
  if(x>0)
    return x;
  else
    return 0;
}

float leaky_relu(float x){//Leaky ReLu
    if(x>0)
      return x;
    else
      return alpha_for_relu*x;
}

float sigmoid(float x){//Sigmoid
    return 1/(1+exp(x*(-1)));
}

float Tanh(float x){//tan hyperbolic
  return (exp(x*2)-1)/(exp(x*2)+1);
}

int main(){
  //take dimensions from users
  cout<<"Enter number of input dimensions:";
  cin>>input_dim;
  cout<<endl;
  cout<<"Enter number of output dimensions:";
  cin>>output_dim;
  cout<<endl;
  cout<<"Enter number of hidden layers:";
  cin>>hidden_layers;
  cout<<endl;
  for(int i = 0; i < hidden_layers; i++){
    cout<<"Enter number of nodes in "<<i+1<<"th hidden layer:";
    int a;
    cin>>a;
    cout<<endl;
    cout<<"Enter activation function for the layer: (1: relu, 2: leaky_relu, 3: sigmoid, 4: tanh)";
    int b;
    cin>>b;
    hidden_dim.push_back(a);
    activations.push_back(b);
    cout<<endl;
  }
  cout<<"Enter activation function for the output layer: (1: relu, 2: leaky_relu, 3: sigmoid, 4: tanh)";
  int b2;
  cin>>b2;

  activations.push_back(b2);
  cout<<endl;
  //Assumed one hidden layer with 5 nodes
  vector<vector<vector<float> > > weight_matrices;
  vector<vector<float> > weight_matrix1;
  //Start initializing weight matrices here
  //weight matrix for input layer and first hidden layer
  for(int i = 0; i < hidden_dim[0]; i++){
    vector<float> v1;
    for(int j = 0; j < input_dim; j++){
      v1.push_back(0);//put some random weights here.Need to discuss this.
    }
    weight_matrix1.push_back(v1);
  }
  weight_matrices.push_back(weight_matrix1);
  //Code here to generalize to more than one hidden hidden_layers
  for(int i=0; i < hidden_layers-1; i++){
    vector<vector<float> > weight_matrix2;
  }
  //weight matrix for last hidden layer and output layer
  vector<vector<float> > weight_matrix3;
  for(int i = 0; i < output_dim; i++){
    vector<float> v2;
    for(int j = 0; j < hidden_dim[hidden_dim.size()-1]; j++){
      v2.push_back(0);//put some random weights here.Need to discuss this.
    }
    weight_matrix3.push_back(v2);
  }
  weight_matrices.push_back(weight_matrix3);
  weight_matrices[0][0][0] = 0.2;
  weight_matrices[0][0][1] = 0.4;
  weight_matrices[0][0][2] = -1*0.5;
  weight_matrices[0][1][0] = -1*0.3;
  weight_matrices[0][1][1] = 0.1;
  weight_matrices[0][1][2] = 0.2;
  weight_matrices[1][0][0] = -1*0.3;
  weight_matrices[1][0][1] = -1*0.2;
  vector<float> input;//vector containing 1 input data
  input.push_back(1);
  input.push_back(0);
  input.push_back(1);
  //read this input from the file
  vector<float> output;//vector containing outputs received at final layer
  vector<vector<float> > hidden_layer_values;
  vector<float> hidden1;
  for(int i = 0; i < weight_matrices[0].size(); i++){
    float sum = 0;
    for(int j = 0; j < input.size(); j++){
      sum = sum+weight_matrices[0][i][j]*input[j];
    }
    cout<<sum<<endl;
    if(activations[0]==1)
      hidden1.push_back(relu(sum));
    else if(activations[0]==2)
      hidden1.push_back(leaky_relu(sum));
    else if(activations[0]==3)
      hidden1.push_back(sigmoid(sum));
    else
      hidden1.push_back(Tanh(sum));
  }
  hidden_layer_values.push_back(hidden1);//hidden layer 1 values


  for(int i = 1; i < weight_matrices.size()-1; i++){//fro 2nd hidden layer to last hidden layer populate values
    vector<float>hidden2;
    for(int j = 0; j < weight_matrices[i].size(); j++){//for each node in the i+1th hidden layer compute the value
      float sum = 0;
      for(int k = 0; k < hidden_layer_values[i-1].size(); k++){//values from previous hidden layer multiplied by their respective weights
        sum = sum+weight_matrices[i][j][k]*hidden_layer_values[i-1][k];
      }
      if(activations[i]==1)
        hidden2.push_back(relu(sum));
      else if(activations[i]==2)
        hidden2.push_back(leaky_relu(sum));
      else if(activations[i]==3)
        hidden2.push_back(sigmoid(sum));
      else
        hidden2.push_back(Tanh(sum));
    }
    hidden_layer_values.push_back(hidden2);
  }
  //vector<float>hidden3;
  for(int i = 0; i < weight_matrices[weight_matrices.size()-1].size(); i++){
    float sum = 0;
    for(int j = 0; j < hidden_layer_values[hidden_layer_values.size()-1].size(); j++){
      sum = sum+weight_matrices[weight_matrices.size()-1][i][j]*hidden_layer_values[hidden_layer_values.size()-1][j];
    }
    cout<<sum<<endl;
    if(activations[activations.size()-1]==1)
      output.push_back(relu(sum));
    else if(activations[activations.size()-1]==2)
      output.push_back(leaky_relu(sum));
    else if(activations[activations.size()-1]==3)
      output.push_back(sigmoid(sum));
    else
      output.push_back(Tanh(sum));
  }
  //output.push_back(hidden3);//output layer values

  for(int i = 0; i < hidden_layer_values[0].size(); i++){
    cout<<hidden_layer_values[0][i]<<" ";
  }
  cout<<endl;
  for(int i = 0; i < output.size(); i++){
    cout<<output[i]<<" ";
  }
  cout<<endl;
}

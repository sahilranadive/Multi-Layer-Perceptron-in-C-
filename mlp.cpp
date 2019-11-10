#include <bits/stdc++.h>
#include <string>

using namespace std;


//declare global variables here
float learning_rate = 0.1;
float alpha_for_relu = 0.05;
int input_dim;
vector<int> hidden_dim;
int output_dim;
int hidden_layers;
vector<int> activations;
vector<vector<vector<float> > > weight_matrices;
vector<vector<vector<float> > > change_in_weights;
int gradient_type;
int batch_size;
//start defining activation functions here

void printVector(vector<float> v){
  for(int i= 0; i < v.size();i++){
    cout<<v[i]<<" ";
  }
  cout<<endl;
}
void read_record(vector<vector<float> > &input,vector<vector<float> >&labels,int labelcount)
{
  // File pointer
  fstream fin;
  // Open an existing file
  fin.open("winequality-red.csv", ios::in);
  string line, word, temp;

  while (getline(fin, line)) {
    vector<float> row;
    // used for breaking words
    stringstream s(line);
    //cout<<line;
    // read every column data of a row and
    // store it in a string variable, 'word'
    while (getline(s, word, ';')) {

      // add all the column data
      // of a row to a vector
      row.push_back(atof(word.c_str()));
    }
    vector<float> label;
    for(int i = 0; i < labelcount;i++){
      label.push_back(row[row.size()-1]);
      row.pop_back();
    }
    printVector(label);
    printVector(row);
    input.push_back(row);

    labels.push_back(label);
  }
}

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

float multiplyingFactor(float x,int n){
  if(n==1){
    if(x<=0)
      return 0;
    else{
      return 1;
    }
  }
  else if(n==2){
    if(x<=0){
      return alpha_for_relu;
    }
    else{
      return 1;
    }
  }
  else if(n==3){
    return x*(1-x);
  }
  else{
    return (1-x*x)/2;
  }
}

vector<vector<float> > makeMatrix(int dim1,int dim2){//returns matrix of size dim2xdim1
  vector<vector<float> > ans;
  for(int i = 0; i < dim2; i++){
    vector<float> v1;
    for(int j = 0; j < dim1; j++){
      v1.push_back(0);//put some random weights here.Need to discuss this.
    }
    ans.push_back(v1);
  }
  return ans;
}

vector<float> intermediateOutputs(vector<vector<float> > weight_matrix,vector<float> input){
  vector<float> hidden1;
  for(int i = 0; i < weight_matrix.size(); i++){
    float sum = 0;
    for(int j = 0; j < input.size(); j++){
      sum = sum+weight_matrix[i][j]*input[j];
    }
    //cout<<sum<<endl;
    if(activations[0]==1)
      hidden1.push_back(relu(sum));
    else if(activations[0]==2)
      hidden1.push_back(leaky_relu(sum));
    else if(activations[0]==3)
      hidden1.push_back(sigmoid(sum));
    else
      hidden1.push_back(Tanh(sum));
  }
  return hidden1;
}

void initializeError(vector<vector<float> > &error_matrix){
  vector<float> v;
  for(int i = 0; i < output_dim; i++){
    v.push_back(0);
  }
  error_matrix.push_back(v);
  for(int i = hidden_layers-1; i >=0; i--){
    vector<float> error;
    for(int j = 0; j < hidden_dim[i]; j++){

      error.push_back(0);
    }
    error_matrix.push_back(error);
  }
}

void populateErrorMatrix(vector<vector<float> >&error_matrix,vector<float> output, vector<vector<float> > hidden_layer_values,vector<float> label){

  for(int i = 0; i < output.size(); i++){
    float e = multiplyingFactor(output[i],activations[activations.size()-1])*(label[i]-output[i]);
    error_matrix[0][i]+=e;

  }

  for(int i = 0; i < hidden_layer_values[hidden_layer_values.size()-1].size(); i++){
    float v = 0;
    for(int j = 0; j < output.size();j++){
      v += output[j]*weight_matrices[weight_matrices.size()-1][j][i];
    }
    float e = multiplyingFactor(hidden_layer_values[hidden_layer_values.size()-1][i],activations[activations.size()-2])*v;
    error_matrix[1][i]+=e;
  }
  //cout<<"starting here"<<endl;
  int weight_matrix_counter = weight_matrices.size()-2;
  for(int i = hidden_layers-2; i >=0; i--){
    for(int j = 0; j < hidden_layer_values[i].size(); j++){
      float v = 0;
      for(int k = 0; k < hidden_layer_values[i+1].size();k++){
        v += hidden_layer_values[i+1][k]*weight_matrices[weight_matrix_counter][k][j];
      }
      float e = multiplyingFactor(hidden_layer_values[i][j],activations[i])*v;
      error_matrix[hidden_layers-i][j]+=e;
    }
  }
  //cout<<"done here"<<endl;
}

void populateChangeInWeightMatrix(vector<vector<float> > &error_matrix,vector<vector<float> >input,int s, vector<vector<float> > hidden_layer_values){
  //int a = hidden_layer_values.size()-1;
  int error_matrix_counter = 0;
  int hidden_layers_counter = hidden_layer_values.size()-1;
  for(int i = weight_matrices.size()-1; i > 0; i--){
    for(int j = 0;j<weight_matrices[i].size();j++){
      for(int k = 0; k < weight_matrices[i][j].size();k++){
        change_in_weights[i][j][k] += learning_rate*error_matrix[error_matrix_counter][j]*hidden_layer_values[hidden_layers_counter][j]/batch_size;
        //cout<<"weight_matrix["<<i<<"]["<<j<<"]["<<k<<"]:"<<change_in_weights[i][j][k]<<endl;
      }
    }
    error_matrix_counter++;
    hidden_layers_counter--;
  }
  //int input_counter = 0;
  //cout<<input[s][0]<<endl;
  //cout<<error_matrix_counter<<" "<<hidden_layers_counter<<" starting here"<<endl;
  //cout<<"weight matrices[0].size(): "<<change_in_weights[0].size()<<endl;
  for(int j = 0;j<weight_matrices[0].size();j++){
    for(int k = 0; k < weight_matrices[0][j].size();k++){
      //cout<<error_matrix[error_matrix_counter][j]<<endl;
      //cout<<input[s][k]<<endl;
      change_in_weights[0][j][k] += learning_rate*error_matrix[error_matrix_counter][j]*input[s][k]/batch_size;
      //cout<<"weight_matrix[0]["<<j<<"]["<<k<<"]:"<<change_in_weights[0][j][k]<<endl;
    }
    //input_counter++;
  }
}

void updateWeights(){
  //int a = hidden_layer_values.size()-1;
  for(int i = 0; i < weight_matrices.size();i++){
    for(int j= 0; j < weight_matrices[i].size();j++){
      for(int k = 0; k < weight_matrices[i][j].size();k++){
        weight_matrices[i][j][k]+=change_in_weights[i][j][k];
      }
    }
  }
}

void printWeights(vector<vector<float> > &error_matrix,vector<float> input, vector<vector<float> > hidden_layer_values){
  for(int i = 0; i < weight_matrices.size();i++){
    for(int j= 0; j < weight_matrices[i].size();j++){
      for(int k = 0; k < weight_matrices[i][j].size();k++){
        cout<<weight_matrices[i][j][k]<<endl;
      }
    }
  }
}
int main(){
  vector<vector<float> > input;//read input from user here
  vector<vector<float> >labels;
  read_record(input,labels,1);

  //vector<float> i1,i2,i3,l1,l2,l3;
  // i1.push_back(1);
  // i2.push_back(0.5);
  // i3.push_back(0.3);
  // input.push_back(i1);
  // input.push_back(i2);
  // input.push_back(i3);
  // l1.push_back(0.8);
  // l2.push_back(0.4);
  // l3.push_back(0.2);
  // labels.push_back(l1);
  // labels.push_back(l2);
  // labels.push_back(l3);

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
  cout<<"Enter type of gradient descnt to be used: (1: Stoichastic, 2: Batch, 3: MiniBatch)";
  cin>>gradient_type;
  cout<<endl;
  if(gradient_type==1){
    batch_size = 1;
  }
  else if(gradient_type==3){
    cout<<"Enter Batch Size: ";
    cin>>batch_size;
    cout<<endl;
  }
  else{
    batch_size = input.size();
  }
  vector<vector<vector<float> > > WeightMatrixInitializer;

  //Start initializing weight matrices here
  weight_matrices = WeightMatrixInitializer;

  weight_matrices.push_back(makeMatrix(input_dim,hidden_dim[0]));
  //Code here to generalize to more than one hidden hidden_layers
  for(int i=0; i < hidden_layers-1; i++){
    weight_matrices.push_back(makeMatrix(hidden_dim[i],hidden_dim[i+1]));
  }
  //weight matrix for last hidden layer and output layer
  weight_matrices.push_back(makeMatrix(hidden_dim[hidden_dim.size()-1],output_dim));

  //Sample weights
  // weight_matrices[0][0][0] = 0.2;
  // weight_matrices[0][0][1] = 0.4;
  // weight_matrices[0][0][2] = -1*0.5;
  // weight_matrices[0][1][0] = -1*0.3;
  // weight_matrices[0][1][1] = 0.1;
  // weight_matrices[0][1][2] = 0.2;
  // weight_matrices[1][0][0] = -1*0.3;
  // weight_matrices[1][0][1] = -1*0.2;
  int samples_processed = 0;
  vector<vector<float> > error_matrix;
  vector<vector<float> > hidden_layer_values;
  while(samples_processed<input.size()){
    vector<vector<float> > error_matrix1;
    error_matrix = error_matrix1;
    hidden_layer_values = error_matrix1;
    initializeError(error_matrix);

    change_in_weights = WeightMatrixInitializer;
    change_in_weights.push_back(makeMatrix(input_dim,hidden_dim[0]));
    //Code here to generalize to more than one hidden hidden_layers
    for(int i=0; i < hidden_layers-1; i++){
      change_in_weights.push_back(makeMatrix(hidden_dim[i],hidden_dim[i+1]));
    }
    //weight matrix for last hidden layer and output layer
    change_in_weights.push_back(makeMatrix(hidden_dim[hidden_dim.size()-1],output_dim));
    for(int i = 0; i < batch_size; i++){
      if(samples_processed==input.size())
        break;
      vector<vector<float> > hiddenValuesInitialzer;
      hidden_layer_values = hiddenValuesInitialzer;

      vector<float> output;//vector containing outputs received at final layer

      hidden_layer_values.push_back(intermediateOutputs(weight_matrices[0],input[samples_processed]));//hidden layer 1 values

      for(int i = 1; i < weight_matrices.size()-1; i++){//fro 2nd hidden layer to last hidden layer populate values
        hidden_layer_values.push_back(intermediateOutputs(weight_matrices[i],hidden_layer_values[i-1]));
      }

      output = intermediateOutputs(weight_matrices[weight_matrices.size()-1],hidden_layer_values[hidden_layer_values.size()-1]);

      populateErrorMatrix(error_matrix,output,hidden_layer_values,labels[samples_processed]);
      cout<<"populated error matrix"<<endl;
      cout<<input[0][0]<<endl;
      populateChangeInWeightMatrix(error_matrix, input,samples_processed, hidden_layer_values);
      //Update weights here
      cout<<"populated change_in_weights matrix"<<endl;
      samples_processed++;

    }
    cout<<samples_processed<<endl;
    updateWeights();

    if(samples_processed==input.size())
      break;
  }
  printWeights(error_matrix, input[samples_processed-1], hidden_layer_values);
}

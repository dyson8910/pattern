#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int argmax(int* array){
  int idx_max = -1;
  int length = sizeof(array)/sizeof(array[0]);
  
  for(int i = 0;i < length;i++){
    if(idx_max < array[i]){
      idx_max = i;
    }
  } 
  return idx_max;
}
  
int main(){
  
  ifstream ifs("iris.data");
  if(!ofs){
    cout << "failed opening iris.data" << endl;
    exit(1);
  }

  
  
  return 0;
}


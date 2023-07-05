#ifndef MSPOAMP_H
#define MSPOAMP_H
#include <iostream>
#include <vector>
#include <cstdlib>
#include <functional>
#include <limits>
#include <algorithm>
#include "Random.h"
#include "InputData.h"

using namespace std;

class MPsoAmp
{
    public:
        MPsoAmp(InputData&);  //constructor
        double c1;              // cognitive learning factor
        double c2;              //social learning factor
        double wIni;            //peso inicial
        double wFim;            //peso final
        double w;               //peso de atualizacao de vetor velocidade V

        Random R;
        // int CountConv;
        InputData Geral;
        vector<double> xMin;    // minumum bound restriction for Independent variable (Optimization) 
        vector<double> xMax;    // maximum bound restriction for Independent variable (Optimization) 
        vector<double> FobjValor; // variavel membro com dim FobjValor(Particulas)
        vector<double> vPl;     // valor do atratores locais - vPl(Particula)
        vector<double> vMax;    // diretamente proporcional a distrancia entre xMin[i] e xMax[i]
        vector<vector<double> > X;  // matriz que guarda "posicao" atual da populacao, ou as variaveis independentes da FObj, X(Particulas, Parametros) 
        vector<vector<vector<double> > > X_dynamic;
        vector<vector<double> > Pl; // Local attractor (individual), Pl(Particulas, Parametros) 
        vector<vector<double> > V;  //matriz contendo o vetor velocidade "imaginaria" das particulas, e usada p/ atualizar X.... V(Particulas, Parametros)
        void EvalObjFun(double (std::vector<double>&), double, InputData&);   // Prototype Declaration das funções membro ou métodos (member functions)
        void EvalSolution(double);   //Atualiza Pg e Pl (os atratores global e local (ou melhor solucao gobal e local))
        void UpdatePosition(double); //Estima o novo vetor de variaveis independentes: // X_k+1 = X_k + Beta(V[i][j],RN)
        void OptMain(double (std::vector<double>&), string&, InputData&); // "()main" da classe

        //MAPSO
        vector<double> vPg, vPgN;
        vector<vector<double>> Pg;
        vector<double> Vc_min, ro1_min;
        vector<vector<double>> Vc, ro1_corr, F;
        vector<vector<double>> omega_zero, c_zero, alpha, Phi_one, Phi_two, m1, m2;
        vector<int> indice_min;

        int nr_of_sub_swarms, nr_of_sub_swarms_complement;

};

#endif // MSPOAMP_H

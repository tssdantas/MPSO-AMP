#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <functional>
#include <limits>
#include <algorithm>
#include "Random.h"
#include "inputData.h"
#include "MPsoAmp.h"
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;


// CLASSE::CONSTRUCTOR 
MPsoAmp::MPsoAmp(InputData &data)  // Constructor (outline) p/ a classe ExameParticulas declarada em ExameParticulas.h 
{
    Geral = data; // transfere os membros do objeto "&data" para o objeto "Geral" do escopo dessa função
    R.SeedInput();

    // ---- Parametros do método enxame de particulas
    c1 = 2.0;       //cognitive learning factor
    c2 = 2.0;       //social learning factor
    //CountConv = 0.0;    // nao e usado
    wIni = 0.9;         // peso inicial para atualizar o vetor de velocidade V
    wFim = 0.4;         // peso final para atualizar o vetor de velocidade V
    //vPg = 0.0;          // valor funcao obj do lider (metodo de otimizacao Enxame de Particulas)

    // ---- Ajusta a dimensao dos vetores FobjValor, vP1, DistBest para o numero (populacao) de particulas 
    w = wIni;
    FobjValor.resize(Geral.particles);  
    vPl.resize(Geral.particles);
    //DistBest.resize(Geral.particles); Nao esta sendo usado...

    for (int i = 0; i < Geral.particles; i++)
    {
    	vPl[i] = 0;                 // inicializa os elementos do vetor vP1 e faz cada elemento = zero .
    }
    
    // ---- Ajusta a dimensão dos vetores Pg[i], vMax[i] igual ao numero de parametros ou variaveis independentes.
    Pg.resize(Geral.nr_independent_variables);    
    vMax.resize(Geral.nr_independent_variables);

    X.resize(Geral.particles);  // dimensiona as linhas da matriz X[i][j] igual ao numero de particulas 
    for (int i = 0; i < Geral.particles; i++)
    {
        X[i].resize(Geral.nr_independent_variables); //  operacao de resize nas colunas da linha i (dimensao de cada linha e independente)
    }

    V.resize(Geral.particles);  // dimensiona as linhas da matriz V[i][j] igual ao numero de particulas 
    for (int i = 0; i < Geral.particles; i++)
    {
        V[i].resize(Geral.nr_independent_variables);  //  operacao de resize nas colunas da linha i (dimensao de cada linha e independente)
    }

    Pl.resize(Geral.particles);   // dimensiona as linhas da matriz P1[i][j] igual ao numero de particulas
    for (int i = 0; i < Geral.particles; i++)
    {
        Pl[i].resize(Geral.nr_independent_variables);  //  operacao de nas colunas da linha i (dimensao de cada linha e independente)
    }

    // Faz os vetores xMax e xMin possuirem a mesma dimensao do numero de parametros.
    xMax.resize(Geral.nr_independent_variables);   
    xMin.resize(Geral.nr_independent_variables);    

    // transfere dados de xMax e xMin do objeto de entrada (Geral) para uma variavel membro da classe MPsoAmp
    for (int j = 0; j < Geral.nr_independent_variables; j++)         
    {                                                  
		xMax[j] = Geral.xMax;
		xMin[j] = Geral.xMin;
    }

    // Estima vMax... proporcional a diferença entre xMin e xMax
    for (int j = 0; j < Geral.nr_independent_variables; j++)        
    {
		vMax[j] = 0.25*(xMax[j] - xMin[j]);
    }

    // ---- Inicializa as matrizes X[i][j] e V[i][j] com numeros randomicos...
    // ... dando as particulas uma posicao inicial X no espaco de busca e um vetor velocidade inicial V.
    #pragma omp parallel for
    for (int i = 0; i < Geral.particles; i++)        
    {
        for (int j = 0; j < Geral.nr_independent_variables; j++)
        {
            //std::uniform_real_distribution<double> dist_uniform_xMaxMin(xMin[j], xMax[j]);

            X[i][j] = xMin[j] + 0.1*R.dRN(xMin[j], xMax[j]);
            V[i][j] = vMax[j]*(2*R.dRN(xMin[j], xMax[j]) - 1);
        }
    }
    // ---------------------- MAPSO ---------------------------- //
    if ( ((double) Geral.particles)/10.0 < 1.0) 
    { 
        nr_of_sub_swarms = 1; 
        cout << "Nr of SubSwarms : " << nr_of_sub_swarms << endl;
        nr_of_sub_swarms_complement = 0;
    } else { 
        //nr_of_sub_swarms = 1; 
        nr_of_sub_swarms = Geral.particles/10;
        nr_of_sub_swarms_complement = Geral.particles - (nr_of_sub_swarms*10);
        cout << "Nr of SubSwarms : " << nr_of_sub_swarms << " nr_of_sub_swarms_complement : " << nr_of_sub_swarms_complement << endl;
    }
    cout << endl << endl;
    

    vPg.resize(nr_of_sub_swarms);
    vPgN.resize(nr_of_sub_swarms);
    indice_min.resize(nr_of_sub_swarms);

    Vc_min.resize(nr_of_sub_swarms);
    ro1_min.resize(nr_of_sub_swarms);

    for (int i = 0; i < nr_of_sub_swarms; i++) 
    { 
        vPg[i] = 0; 
        vPgN[i] = 0;
        indice_min[i] = 0;
    } 
    
    for (int i = 0; i < Geral.nr_independent_variables; i++) { Pg[i].resize(nr_of_sub_swarms); }

    int t1 = 0.20*Geral.iterations; int t2 = 0.80*Geral.iterations;
    double Vc_max = 40;  
    double ro1_max = 0.8;
    double F_max = 15; double F_min = 0.25;

    for (int i = 0; i < nr_of_sub_swarms; i++)  
    { 
        // These are ajustable paramerters for the movement pattern adaptation heuristic ! (8.0, 10.0, 0.3 and 0.5)
        Vc_min[i] = R.dRN(8.0, 10.0);
        ro1_min[i] = R.dRN(0.3, 0.5);
    } 

    Vc.resize(Geral.iterations+1); 
    ro1_corr.resize(Geral.iterations+1);
    F.resize(Geral.iterations+1);
    for (int i = 0; i < Geral.iterations+1; i++) 
    {
        Vc[i].resize(nr_of_sub_swarms);
        ro1_corr[i].resize(nr_of_sub_swarms);
        F[i].resize(nr_of_sub_swarms);
    }

    #pragma omp parallel for
    for (int tt = 0; tt < nr_of_sub_swarms; tt++) 
    {
        for (int t = 0; t < Geral.iterations+1; t++) 
        {

            if (t < t1)
            {
                Vc[t][tt] = Vc_max;
                ro1_corr[t][tt] = ro1_min[tt];
                F[t][tt] = F_min;
            } else if (t >= t1 && t < ((t2-t1)/2) + t1) {

                Vc[t][tt] = ((t - t1)*(Vc_min[tt] - Vc_max))/(t2 - t1) + Vc_max;
                ro1_corr[t][tt] = ( ((t - t1)*(ro1_max - ro1_min[tt]))/((t2 - t1)/2) ) + ro1_min[tt];
                F[t][tt] = 1;

            } else if ( t >= ((t2-t1)/2) + t1 && t < t2 ) {

                Vc[t][tt] = ((t - t1)*(Vc_min[tt] - Vc_max))/(t2 - t1) + Vc_max;
                ro1_corr[t][tt] = ( ((t - t2)*(ro1_min[tt] - ro1_max))/((t2 - t1)/2) ) + ro1_min[tt];
                F[t][tt] = 1;
            
            } 
            else if (t >= t2) 
            {
                Vc[t][tt] = Vc_min[tt];
                ro1_corr[t][tt] = ro1_min[tt];
                F[t][tt] = F_max;

            }
        }
    }
 
    m1.resize(Geral.iterations+1);
    m2.resize(Geral.iterations+1);
    alpha.resize(Geral.iterations+1);
    omega_zero.resize(Geral.iterations+1);
    c_zero.resize(Geral.iterations+1);
    Phi_one.resize(Geral.iterations+1);
    Phi_two.resize(Geral.iterations+1);

    for (int i = 0; i < Geral.iterations+1; i++) 
    {
        m1[i].resize(nr_of_sub_swarms);
        m2[i].resize(nr_of_sub_swarms);
        alpha[i].resize(nr_of_sub_swarms);
        omega_zero[i].resize(nr_of_sub_swarms);
        c_zero[i].resize(nr_of_sub_swarms);
        Phi_one[i].resize(nr_of_sub_swarms);
        Phi_two[i].resize(nr_of_sub_swarms);
    }

    #pragma omp parallel for
    for (int tt = 0; tt < nr_of_sub_swarms; tt++) 
    {
        for (int t = 0; t < Geral.iterations+1; t++) 
        {

            alpha[t][tt] = sqrt(F[t][tt]);
            m1[t][tt] = (1 + 3*alpha[t][tt] + pow(alpha[t][tt], 2))*(pow( (alpha[t][tt] + 1) , 2));
            m2[t][tt] = (2 + 3*alpha[t][tt] + 2*pow(alpha[t][tt], 2))*(pow( (alpha[t][tt] + 1) , 2));
            omega_zero[t][tt] = (m1[t][tt]*Vc[t][tt] + m2[t][tt]*ro1_corr[t][tt]*Vc[t][tt] + ro1_corr[t][tt] - 1 )/( m2[t][tt]*Vc[t][tt] + m1[t][tt]*ro1_corr[t][tt]*Vc[t][tt] - ro1_corr[t][tt] + 1 );
            c_zero[t][tt] = (2*(1 - ro1_corr[t][tt])*(omega_zero[t][tt] + 1))/(1 + alpha[t][tt]);
            Phi_one[t][tt] = (2*c_zero[t][tt])/(1 + alpha[t][tt]);
            Phi_two[t][tt] = (2*alpha[t][tt]*c_zero[t][tt])/(1 + alpha[t][tt]);
        }
    }
         

    //Particle pattern of oscilation study
    X_dynamic.resize(Geral.iterations+1);   
    for (int i = 0; i < Geral.iterations+1; i++)
    {
        X_dynamic[i].resize(Geral.nr_independent_variables); 
        for (int j = 0; j < Geral.nr_independent_variables; j++)
        {
            X_dynamic[i][j].resize(Geral.particles);
        }

    }

    // Transferindo o enxame inicial para o vetor de histórico
    for (int i = 0; i < Geral.particles; i++)        
    {
        for (int j = 0; j < Geral.nr_independent_variables; j++)
        {
            X_dynamic[0][j][i] = X[i][j];
        }
    }

}

// CLASSE::FUNCAO.MEMBRO(ARGs)
void MPsoAmp::EvalObjFun(double (*f)(std::vector<double>&), double k, InputData &data1) 
{
    //#pragma omp parallel for  // habilita a biblioteca OpenMP para programação paralela em multiplos threads
    #pragma omp parallel for
	for (int i = 0; i < Geral.particles; i++)
	{
        // FobjValor é variavel (vetor) membro que armazena os valores da Fobj
        
        FobjValor[i] = f(X[i]);  

        // double f(vector<double> é uma prototype declaration da funcao obj que foi declarada no escopo global
        // em main.cpp. Nesse caso, o compilador interpreta essa prototype declaration e transforma em 
        // um pointer apropriado.
	}
}

// CLASSE::FUNCAO.MEMBRO(ARGs)
void MPsoAmp::EvalSolution(double k)
{
    ///////////////////////////////////////////////////////////////////////////////////
    // ------- Atualiza Pg e Pl (os atratores global e local (melhor sol.))  --------//
    // -----  Calcula vPgN (populacao atual), vPg (global todas iter) p/ pG     -----//

    // NEW ------------------------------------------------------------------------------
    

    // Grava em vPgN(double) o valor minimo da funcao obj da populacao ou particulas em (FobjValor), nessa iteracao k.
    // ...
    // sobre o min_elemen: "returns an iterator pointing to the element with the smallest value in the range;  () e end() are iterators of the vector classe" 
    
    #pragma omp parallel for
    for ( int tt = 0; tt < nr_of_sub_swarms; tt++)
    {
        int complement = 0;
        if (tt == (nr_of_sub_swarms - 1) ) { complement = nr_of_sub_swarms_complement; } 
        else if (tt < (nr_of_sub_swarms - 1)) { complement = 0;} 

        if (nr_of_sub_swarms == 1) 
        {
            vPgN[0] = *min_element(FobjValor.begin(), FobjValor.end());

        } else if (nr_of_sub_swarms > 1 ) {

            vPgN[tt] = *min_element(FobjValor.begin() + 10*tt, FobjValor.begin() + 10*(tt + 1) + complement );
        }


        vector<double>::iterator it_min, it_max; //declara um iterador "it" para um objeto do tipo vector<double>
        it_min = find(FobjValor.begin() + 10*tt, FobjValor.begin() + 10*(tt+1) + complement, vPgN[tt]);   //acha a posicao de vPgN no vetor FobjValor

        // Transforma a distancia entre o iterador inicial e o iterador de vPgN em variavel do tipo int ou integer.
        indice_min[tt] = distance(FobjValor.begin(), it_min); 

        // ---- Verifica na populacao atual na iteracao k, ha valor da func.obj vPgN=f(FobjValor[i]) menor em comparacao com o minima global em vPg, que foi obtido nas iteracoes ate k-1.
        // ---- Transfere os parametros de X[i][j], econtrados na iteracao k, para Pg[i] da nova solucao global em vPgN=f(FobjV[i]) .
        if (vPgN[tt] < vPg[tt] )     
        {

            vPg[tt] = vPgN[tt];
            for (int j = 0; j < Geral.nr_independent_variables; j++)
            {
                Pg[j][tt] = X[indice_min[tt]][j];  // X(particulas,parametros)
            }
        
        }
        else if (vPg[tt] == 0)      // Na primeira iteracao (vPg == 0) ... 
        {                       //.. entao o codigo atualiza o Pg[i] com a melhor econtrada em FobjValor[] da interacao k = 1.
            vPg[tt] = vPgN[tt];
            for (int j = 0; j < Geral.nr_independent_variables; j++)
            {
                Pg[j][tt] = X[indice_min[tt]][j];
            }
        }
    }



    // ---- Verifica na populacao atual ha valor da funcao objetivo FobjValor[i] menor em comparacao com aos minimos locais em vPl[i] = f(Pl[i][j]),  
    // ---- Transfere os parametros dos novos atratores locais econtrados em X[i][j] na iteracao k, para o termo referente ao atrator local Pl[i][j] das futuras iteracoes k + n, onde n = 1,2,...
    #pragma omp parallel for
    for (int i = 0; i < Geral.particles; i++)
    {
    // Atualiza os atratores locais se FobjValor[i] (interacao k) e menor que vPl[i] (interacao k-1)
        if ( FobjValor[i] < vPl[i] )
        {
            vPl[i] = FobjValor[i];

            for (int j = 0; j < Geral.nr_independent_variables; j++)
            {
				Pl[i][j] = X[i][j];
            }
        }
        else if (vPl[i] == 0)     //vPl[i] == 0 implica que na primeira iteracao o atrator local Pl = posicao inicial X   
        {                         // nas iteracoes futuras (k + n para n = 1,2,..) vPl é comparado com FobjValor[i] como foi feito acima.
            vPl[i] = FobjValor[i];
            for (int j = 0; j < Geral.nr_independent_variables; j++)
            {
				Pl[i][j] = X[i][j];
            }
        }
    }

    // Na convergencia, iguala-se os atratores globais de cada enxame
    if ( k >= 0.8*Geral.iterations ) 
    {
        double vPg_BestSwarm = *min_element(vPg.begin(), vPg.end());
        vector<double>::iterator it_BestSwarm; //declara um iterador "it" para um objeto do tipo vector<double>
        it_BestSwarm = find(vPg.begin(), vPg.end(), vPg_BestSwarm);   //acha a posicao de vPgN no vetor FobjValor
        int indice_BestSwarm = distance(vPg.begin(), it_BestSwarm); 
        for (int ii = 0; ii < nr_of_sub_swarms; ii++) 
        {
            if (ii != indice_BestSwarm ) 
            {
                vPg[ii] = vPg[indice_BestSwarm];  
                for (int j = 0; j < Geral.nr_independent_variables; j++)
                {
                    Pg[j][ii] =  Pg[j][indice_BestSwarm]; 
                }
            }
        }

    }
}
// -----           Fim atualizacao atratores Pg[i] e Pl[i][j]               ------//
///////////////////////////////////////////////////////////////////////////////////


// CLASSE::FUNCAO.MEMBRO() 
// Usa a equacao caracteristica dos metodos de otimizacao heursticos, y_k+1 = y_k + c*RandomN, para explorar a regiao de busca.
void MPsoAmp::UpdatePosition(double k)
{
    int iteration = k;
    int index_swarm = 0;
    #pragma omp parallel for
	for (int i = 0; i < Geral.particles; i++)
	{
        if ( i > 1 && (i % 10 == 0) && index_swarm < nr_of_sub_swarms - 1) {index_swarm++;} // incrementa index_swarm

        //cout << "index : " << index_swarm << endl; 

		for (int j = 0; j < Geral.nr_independent_variables; j++)
		{
            // ---- Atualiza o vetor velocidade V da populacao (estocastico) V_k+1 = V_k + c*RandomN*Beta(Pl,Pg)
			V[i][j] = V[i][j]*omega_zero[iteration][index_swarm] + Phi_one[iteration][index_swarm]*R.dRN(0.0, 0.999)*(Pl[i][j] - X[i][j]) + Phi_two[iteration][index_swarm]*R.dRN(0.0, 0.999)*(Pg[j][index_swarm] - X[i][j]); 

            // ---- Controle para que V_k+1 nao utrapasse as restricoes vMax e vMin
			if (fabs(V[i][j]) > vMax[j])   // std::fabs retorna o valor absoluto de arg1
			{
				if (V[i][j] < 0 )       
				{
					V[i][j] = -vMax[j];
				}
				else
				{
					V[i][j] = vMax[j];
				}
			}

            // X_k+1 = X_k + Alpha*S
            //funcao principal do metodo exame de particulas, atualiza a posicao X(Particulas, Params) das variaveis indepentes
			X[i][j] = X[i][j] + V[i][j];  

            // ---- Controle para que X_k+1 nao utrapasse as restricoes xMax e xMin
			if (X[i][j] > xMax[j] )   // se x[i][j] for maior que o seu limite maximo, fazer x = xMax
			{
				X[i][j] = xMax[j];
			}
			else if (X[i][j] < xMin[j])
			{
				X[i][j] = xMin[j];  // se x[i][j] for maior que o seu limite minimo, fazer x = xMin
			}
            
            // guardando em X_dynamic
            X_dynamic[iteration][j][i] = X[i][j];

		}
	}
}

// CLASSE::FUNCAO.MEMBRO(ARGs) 
// mais ou menos equivalente ao ()main da classe ... invoca todas as outras funcoes membro sequencialemente.
// ..
// double f(vector<double> é uma prototype declaration da funcao obj que foi declarada no escopo global
// em main.cpp. Nesse caso, o compilador interpreta essa prototype declaration e transforma em 
// um pointer  ou ponteiro apropriado.
// Equivale a: //void MPsoAmp::Otimizar(double (*f)(vector<double>, InputData&), string  &ArquivoSaida)
void MPsoAmp::OptMain(double (*f)(std::vector<double>&), string  &ArquivoSaida, InputData &data2)
{
    ofstream fout3(ArquivoSaida);  
    
    double k = 0;   //indica p/ funcao membro MPsoAmp::AvaliarSolucao se e ou nao a primeira iteracao.
    EvalObjFun(f, k, data2); // passa o valor da funcao-argumento (double f(vector<double>) para a variavel membro FobjValor(i)
    EvalSolution(k); // inicializa os os atratores locais e global

	// ---- loop de otimizacao
    for (int i = 0; i < Geral.iterations; i++)
    {
    	k++;; //indica p/ funcao membro MPsoAmp::AvaliarSolucao se e ou nao a primeira iteracao.
        UpdatePosition(k); // atualiza a posicao da populacao no espaco de busca
        EvalObjFun(f, k, data2);  // passa o valor da funcao-argumento (double f(vector<double>) para a variavel membro FobjValor(i)
        EvalSolution(k); // atualiza os atratores locais e global
        // O peso w e atualizado ao longo da resolucao, favorecendo varredura global do espaco de busca no inicio,
        // e a convergencia das particulas do exame (populacao) para a solucao global econtrada pelo lider (ou guia) no fim.
        w = wFim + ((((double) Geral.iterations) - ((double) i) )/Geral.iterations)*(wIni - wFim);     

        //apresenta alguns resultados ao usuario
        cout <<  "   Iteration : [" << i << "]   ";
        for (int tt = 0; tt < nr_of_sub_swarms; tt++)
        {
            cout << " Obj.Fun.Swarm (" << tt << ") : " << vPg[tt] << " ";
        }
        cout << endl;
    }
    cout << endl;


    for (int tt = 0; tt < nr_of_sub_swarms; tt++)
    { 
        cout << "Objective Function (leader) vPg of swarm: "<< tt << " = " << vPg[tt] << endl;
        cout << "Independent Variables of swarm " << tt <<  " : " << endl;    
        for (int i = 0; i < Geral.nr_independent_variables; i++)
        {
		    cout << "x[" << i << "] : " << Pg[i][tt] << endl;   //solucao global (lider)
        }
        cout << endl;
    }

}



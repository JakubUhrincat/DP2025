# Použitie:
Pre použitie modulu klasifikácie na princípe prenosu tepla:

    python heat_transfer_method.py

Pre použitie modulu systému viacerých klasifikátorov:

    python classifiers_method.py


# Prvotné spustenie:
Pred prvým spustením je nutné spustiť tento príkaz pre inštaláciu potrebných balíčkov:

    pip install -r requirements.txt


# Opis:
Tento projekt kombinuje viacero prístupov ku paralelnej klasifikácii dát pomocou bunkových automatov. Obsahuje dva moduly, ktoré vykonávajú klasifikáciu spolu s vizualizáciou. Prvý modul simuluje proces šírenia tepla pre účely klasifikácie. Druhý modul používa viacero klasifikátorov pre kombináciu výsledkov viacerých modelov pre následnú predikciu. Tento projekt bol vytvorený za účelom diplomovej práce.


# Obsah repozitára:
Tento repozitár pozostáva z:

**heat_transfer_method.py** – kód modulu prvej metódy na princípe prenosu tepla

**classifiers_method.py** – kód modulu druhej metódy systému viacerých klasifikátorov

**iris.csv** - csv súbor datasetu Iris

**breast.csv** - csv súbor datasetu Breast Cancer Wisconsin

**glass.csv** - csv súbor datasetu Glass

**requirements.txt** - textový súbor, ktorý obsahuje verzie balíčkov pre prvotnú inštaláciu

**output*** - priečinok s ukážkou výstupov programu v grafoch a .png formáte

##### *Súbory s označením * nemajú konštantné dáta. Tieto súbory sa menia po spustení alebo ukončení programu.*


# Príklady použitia:
## 1. Modul klasifikácie na princípe prenosu tepla

Pre správne použitie modulu je potrebné nasledujúci príkaz:

        python heat_transfer_method.py
        
  Po spustení sa spustia všetky fázy modulu - trénovanie, testovanie a klasifikáciu pre nové dátové inštancie. Po úspešnom ukončení práce modulu sú všetky grafy dostupné v priečinku output s názvom príslušného datasetu. Okrem výstupných grafov je zobrazený aj konzolový výstup, ktorý obsahuje samostatný výpis pre každú fázu.

  #### Príklad výstupu:
  
          === Training phase ===
          
          Training graph succesfully saved to png.
          
          
          === Testing phase ===
          
          Confusion Matrix:
          
          [[19  0  0]
          
          [ 0  8  5]
          
          [ 0  1 12]]
          
          
          Accuracy: 86.67%
          
          
          Class Iris-setosa - Precision: 100.00%, Recall: 100.00%, F1 Score: 100.00%
          
          Class Iris-versicolor - Precision: 88.89%, Recall: 61.54%, F1 Score: 72.73%
          
          Class Iris-virginica - Precision: 70.59%, Recall: 92.31%, F1 Score: 80.00%
          
          Macro F1 Score: 84.24%
          
          
          === Prediction phase ===
          
          Predicted class: Iris-setosa
          
          Class results: {'Iris-setosa': 100.0, 'Iris-versicolor': 31.02, 'Iris-virginica': 28.35}

Pri module na princípe prenosu tepla záleži aj na hodnotách parametrov m, portion_percentage, range_percentage.

Tieto parametre sa nastavujú priamo v kóde a pre ich zmenu je potrebné upraviť nasledujúce riadky:

    m = hodnota
    
    range_percentage = hodnota
    
    portion_percentage = hodnota


 ## 2. Modul systému viacerých klasifikátorov

Pre použitie tohto modulu je potrebné použiť príkaz:

  python classifiers_method.py

Po spustení sa rovnako spustia všetky fázy modulu. Modul teda prechádza trénovaním, testovaním a klasifikáciou pre nové dáta. Po úspešnom ukončení modulu sa uložia grafy do priečinku output. S grafmi je výstup dopĺňaný aj výpisom do konzoly.

#### Príklad výstupu:

          === Training phase ===
          
          Final CA Grid (Energies):
          
          [[13.4 13.4 13.4 13.4 13.4 13.4 13.4]
          
           [15.8 14.3 14.8 15.8 14.3 15.8 14.8]
           
           [15.8 14.3 15.3 15.8 14.3 15.8 15.3]]
          
           
          === Testing phase ===
          
          Testing graph succesfully saved to png.
          
                   
          === Prediction phase ===
          
          DECISION TREE:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 100.0%, Iris-versicolor: 0.0%, Iris-virginica: 0.0%
          
          
          KNN:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 100.0%, Iris-versicolor: 0.0%, Iris-virginica: 0.0%
          
          
          SVM:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 96.27%, Iris-versicolor: 2.43%, Iris-virginica: 1.3%
          
          
          RANDOM FOREST:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 100.0%, Iris-versicolor: 0.0%, Iris-virginica: 0.0%
          
          
          NAIVE BAYES:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 100.0%, Iris-versicolor: 0.0%, Iris-virginica: 0.0%
          
          
          ADABOOST:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 39.09%, Iris-versicolor: 34.14%, Iris-virginica: 26.77%
          
          
          MLP:
          
          Predicted class is: Iris-setosa
          
          Confidence scores: Iris-setosa: 99.71%, Iris-versicolor: 0.29%, Iris-virginica: 0.0%
          
          
          === Final Prediction ===
          
          Predicted class: Iris-setosa (Confidence: 90.72%)

 ### Načítanie vlastných dát
V prípade načítania vlastných dát je potrebné mať dataset priamo v priečinku, ktorý obsahuje oba moduly. Tento krok zabezpečí bezproblémové načítanie vlastného datasetu. Pre načítanie je potrebné upraviť priamo kód a to nasledujúci riadok:

    data_path = "názov_datasetu.csv"
  
Okrem načítania nového datasetu pre použitie nových dát pre trénovanie a testovanie modulu je možné načítať iba jednu inštanciu dát, ktorá bude následne použitá na poslednú fázu modulov a to na klasifikáciu pre nové dátové inštancie. Dátové inštancie pre predikciu sa definujú v kóde pomocou atribútu X_new.

Príkladom pre definíciu novej inštancie dát pre následnú predikciu môže byť:

    Dataset Iris: 
    
     X_new = np.array([[5.1, 3.5, 1.4, 0.2]])  # Expected: 'Iris-setosa'
  
  
    Dataset Breast Cancer Wisconsin:
   
     X_new = np.array([[5, 3, 3, 1, 2, 1, 3, 1, 1]])  # Expected: 2
  
     
    Dataset Glass:
    
     X_new = np.array([[1.489, 13.3, 4.2, 1.1, 69.0, 0.0, 8.7, 0.0, 0.0]]) # Expected: Class 3


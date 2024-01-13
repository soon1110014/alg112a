
# 基因演算法
## 一、簡介

### 背景：

基因演算法起源於約翰·霍蘭德及其密西根大學團隊於20世紀60年代對細胞自動機的研究。在20世紀80年代初，基因演算法仍然主要處於理論階段，直到匹茲堡舉辦了第一屆世界基因演算法大會。隨著電腦計算能力的提升和實際需求的增加，基因演算法逐漸轉向實際應用。1989年，紐約時報介紹了第一個商業用途的基因演算法，稱為"進化者"（Evolver）。此後，基因演算法在許多領域得到廣泛應用，被財富雜誌500強企業用於時間表安排、資料分析、未來趨勢預測、預算等，解決了眾多組合最佳化問題。

### 基因演算法則：

* 基本精神在於模仿生物界物競天擇、優勝劣敗的自然進化法則。
* 基因演算法則三個主要運算子為 **複製(reproduction)** 、**交配(cover)**以及**突變(mutation)**。
* 應用基因演算法則來解最佳化其基本精神為：將所要搜尋的所有**參數編碼**稱為染色體(chromosonl)的離散(discrete)或二元(binary)字串(string)來代表參數的值，**編碼空間中，會以編碼的形式來代表一個解，就如背景介紹所提，在 GA 通常一個染色體 (Chromosome) 代表求解問題的一個潛在解 (Potential solution)**；如此隨機地重覆產生N個初始物種(字串)，然後依據求解之條件來設計適應函數(fitness function)，適應函數值高的物種將被挑選至交配池(mating pool)中，此即複製過程，再依交配及突變過程的運算，即完成一代的基因演算法則，如此重覆下去以產生適應性最強的物種。

### 流程圖：
![image](https://github.com/soon1110014/alg112a/assets/105185732/a88d88e0-3e8a-46fc-bd0b-582168789a79)

## 二、基本原理

### 複製(reproduction)
* 複製是依據每一物種的適應程度來決定其在下一子代中應被淘汰或複製的個數多寡的一種運算過程，其中適應程度的   量測則是由適應函數來反應。
* (a)**輪盤式選擇**：在每一代的演化過程中，首先依每個物種(字串)的適應函數值的大小來分割輪盤上的位置，適應函數值越大則在輪盤上佔有的面積也越大，每個物種在輪盤上所佔有的面積比例也就代表其被挑選至交配池的機率；然後隨機地選取輪盤上的一點，其所對應的物種即被選中送至交配池中。
* (b)**競爭式選擇**：在每一代的演化過程中，首先隨機地選取兩個或更多個物種(字串)，具有最大適應函數值的物種即被選中送至交配池中。

### 交配(crossover)
* 交配過程是隨機地選取交配池中的兩個母代物種字串，並且彼此交換位元資訊，進而組成另外兩個新的物種。
* 交配過程發生的機率由交配機率所控制。
* 而交配過程有三種型式：**(a)單點交配、(b)兩點交配、及(c)字罩交配。**

### (a)單點交配
* 在所選出的兩字串中，隨機地選取一交配點，並交換兩字串中此交配點後的所有位元。
![image](https://github.com/soon1110014/alg112a/assets/105185732/068919e8-7d6b-4ca2-b330-a10db4b36f15)

### (b)兩點交配
* 在所選出的兩字串中，隨機地選取兩個交配點，並交換兩字串中兩個交配點間的所有位元。 
![image](https://github.com/soon1110014/alg112a/assets/105185732/20dc0740-4d39-4f5d-a262-c2516a9dbc41)

### (c)字罩交配
* 首先產生與物種字串長度相同的字罩當作交配時的位元指標器，其中字罩是隨機地由 0 與 1 所組成，字罩中為 1 的位元即是兩物種字串彼此交換位元資訊的位置。
![image](https://github.com/soon1110014/alg112a/assets/105185732/dfad9103-3ea4-46df-954f-c542b666bf25)

### 突變(mutation)
* 突變過程是隨機地選取一物種字串並且，隨機地選取突變點，然後改變物種字串裡的位元資訊。
* 突變過程發生的機率由突變機率所控制。
* 突變過程可以針對單一位元、或對整個字串進行突變運算、或以字罩突變方式為之。
* 對於二進制的位元字串而言就是將字串中的 0 變成 1， 1 變成 0。
![image](https://github.com/soon1110014/alg112a/assets/105185732/3c7fc6d9-1f03-4c29-b5e1-993f7b78add7)

## 主要特性
* 基因演算法則是以參數集合之編碼進行運算而不是參數本身，因此可以跳脫搜尋空間分析上的限制。
* 基因演算法則同時考慮搜尋空間上多個點而不是單一個點，因此可以較快地獲得整體最佳解 (global optimum)，同時也可以避免陷入區域最佳值 (local optimum)的機會，此項特性是基因演算法則的最大優點。

* **參數設定** : (1)編碼範圍 , (2)字串長度 , (3)族群大小 , (4)交配機率 , (5)突變機率 , (6)適應函數之設計
1. 字串長度 : 長度越長則精準度越高，但所須的編碼、解碼運算也相對增加。 
2. 交配機率 : 交配率越高，則新物種進入族群的速度越快，整個搜尋最佳值的速度也越快。 
3. 突變機率 : 突變是一項必須的運算過程，因為在複製及交配過程中可能使得整個族群裡，所有字串中的某一特定位元皆一樣。 
4. 避免陷入區域最佳值 : ；須視所搜尋空間的維度及參數範圍大小與編碼時所採用的精確度(字串長度)一起考量。 
5. 適應函數之設計原則 : 原則上，適應函數須能反應出不同物種間適應程度的差異即可 。
6. 搜尋終止之條件 : 對於某些線上即時系統而言；為了結省時間，當適應函數值到達系統要求後即可終止搜尋程序。
* **編碼及解碼過程** :假設受控系統中有三個參數要編碼，三個參數值均界於 [0,1] 之間，且每個參數使用五個位元加以編碼，則二進制字串編碼流程如下： 
     隨機設定三個變數值於 [0,1] ，假設：
![image](https://github.com/soon1110014/alg112a/assets/105185732/95df3b5c-5068-430a-9af8-1e75c699a3b0)

     則編碼為： 
![image](https://github.com/soon1110014/alg112a/assets/105185732/512c4feb-f6e9-4256-b797-1c3d178b899b)

     最後的字串為： 00001 00010 00100. 而解碼流程則以反對順序即可完成。

## 適應函數的調整

* 有可能在前面幾次演化過程中，已出現一些表現特別優良的染色體，根據複 製的原理，這些特別優良的染色體會被大量地複製於交配池中，導致染色體的多樣性 (diversity)降低。
* 為了避免上述兩種現象，適應函數的調整變得頗為重要，目前較常使用的適應函數調整法有以下三種：**線性調整(Linear scaling) 、Sigma截取(Sigma truncation) 、乘冪調整(Power law scaling)**
### 線性調整
* 我們將適應函數值依照下列線性方程式予以轉換： 
![image](https://github.com/soon1110014/alg112a/assets/105185732/d2c32a64-8824-48d2-b471-2494125a5b81)

* 我們選擇參數 a 與 b 使得：
![image](https://github.com/soon1110014/alg112a/assets/105185732/6da62387-fe06-4a24-8ce4-0bde5b09a0ac)
![image](https://github.com/soon1110014/alg112a/assets/105185732/4b147252-416c-42f8-be7d-36336161c579)
* 其中是調整後最大適應函數值與調整前的平均適應函數值的倍數。
![image](https://github.com/soon1110014/alg112a/assets/105185732/8f4caab2-bdb3-4a3a-97fa-b919246f32f7)

### Sigma截取
* Forrest 建議使用族群適應函數值的變異數資訊，在做適應函數的調整前先予以前處理。其作法是將調整前的適應函數值依下列式子減去一常數值：
![image](https://github.com/soon1110014/alg112a/assets/105185732/76932aac-d43e-4e83-90c7-b101bf44c500)
其中參數 c 的選擇是族群適應函數值的變異數的一個倍數。「Sigma截取」可以避免調整過後的適應函數值會產生負值。
### 乘冪調整
* 使用乘冪的方式來調整適應函數值，使得調整過後的適應函數值是調整前的適應函數值的乘冪，如下所示：
![image](https://github.com/soon1110014/alg112a/assets/105185732/9bf42f52-b548-42c1-8f01-062486855b49)
## 程式
```
def perform_crossover_operation(self):
self.shuffle_index (self•pop_size)
child_index = self•pop_size child2_
_index = self-pop_
_size+1
count_of_crossover = int(self-crossover_size/2)
for i in range(count_of_crossover): parent1_index = self•indexs[i]
parent2_
index = self. indexs[i+1]
if(self.crossover_type == CrossoverType-Fantia]MarpedCrossoven)
self. Pantial MannedCrossoven(parent1_index, parent2_index, child1_index, child_index)
self.objective_values[child1_index] = self.compute_objective_value(self-chromosomes[ childi
_index])
self.objective_values[child2_index] = self. compute_objective_value(self. chromosomes[child2_index])
childl_index +=2
child2_index +=2

def perform_mutation_operation(self):
self.shuffle_index（self.pop_sizetself.crossover_size）
child1.
_index = self-pop_size+self.crossover_size
for i in range(self.mutation_size):
if(self.mutation_type==MutationType.Inversion):
parent1]
index = self.indexs[1]
self.inversion_mutation(parent1
index, childi.
_index)
self.objective_values[child1_ _index]
= self.compute_objective_value(self.chromosomes[child1_index])
child_index += 1

def evaluate_fitness(self):
for i, chromosome in enumerate(self.chromosomes: self-pop_size]):
self.objective_values[i] = self.compute_objective_value(chromosome)
min_obj_val = np.min（self.objective_values）
max_obj_val = np. max(self.objective_values)
range_obj_val = max_obj_val-min_obj_val
for i,obj in enumerate(self.objective_values):
self.fitness[i] = max(self.least_fitness_factor*range_obj_val,pow(10,-5))+
(max_obj_val-obj)

def update_best_solution (self):
best_index = np.argmax(self.fitness)
if(self.best_fitness<self. fitness[best_index]):
self.best_fitness = self.fitness[best_index]
for i,gene in enumerate(self. chromosomes[best_index]):
self. best_chromosome[i] = gene

def perform_selection (self):
if self.selection_type == SelectionType.Deterministic:
index = np. argsort (self.fitness)[::-1]
elif self.selection_type == SelectionType.Stochastic;
index = [self.do_roulette_wheel_selection(self.fitness) for i in range(self-pop_size)]
else:
index = self.shuffle_index(self. total_size)
for i in range(self-pop_size): for j in range(self-number_of.
_genes):
self.selected_chromosomes[i][j] =
self.chromosomes[index[i]][j]
for i in range(self.pop_size):
for j in range(self.number_of_genes)
self.chromosomes[i][j] = self.selected
_chromosomes[i][j]
```
## 測試結果
```
iteration 0 :
[0 3 6 5 2 4 1 7]：89.30000000000001
iteration 10 :
[5 3 2 0 1 4 6 7]: 56.49999999999999
iteration 20 :
[5 3 6 0 4 1 2 7]: 48.6
iteration 30 :
[5 3 2 0 6 4 1 7]：44.1
iteration 40 :
[5 3 2 0 6 4 1 7]: 44.1
iteration 50 :
[5 3 2 0 6 4 1 7]: 44.1
iteration 60 :
[5 3 2 0 4 6 1 7]: 41.0
```
## 參考資源
1. [基因演算法維基](https://zh.wikipedia.org/zh-tw/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95)
2. [程式碼與原理參考](https://medium.com/qiubingcheng/%E4%BB%A5python%E5%AF%A6%E4%BD%9C%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E4%B8%A6%E8%A7%A3%E6%B1%BA%E5%B7%A5%E4%BD%9C%E6%8C%87%E6%B4%BE%E5%95%8F%E9%A1%8C-job-assignment-problem-jap-b0d7c4ad6d0f)
3. 書籍(類神經網路以及基因演算法則)

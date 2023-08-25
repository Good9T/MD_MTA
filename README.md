# MD_MTA
The source code of MD-MTA
We provide codes for two VRP variants<br>
- Multi-depot Vehicle Routing Problem (MDVRP) <br>
- Multi-depot Open Vehicle Routing Problem (MDOVRP) <br>

### Basic Usage
To train a model
```bash
python MDVRP__Train.py
python MDOVRP__Train.py
```
To test a model
```bash
python MDVRP__Eval.py
python MDOVRP__Eval.py
```
All .py file contain parameters you can modify. <br>
### Used Libraries
python v3.9 <br>
torch==1.10 <br>


# Zhvillimi i një Modeli parashikues të smogut në Kosovë (Prishtinë, Prizren, Pejë)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Phase%201%20Done-success)
<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master </p>
      <p><strong>Profesorët:</strong> Prof. Dr. Lule Ahmedi, PhD Mergim Hoti </p>
      <p><strong>Lënda:</strong> “Machine Learning”</p>
      <p><strong>Grupi 5:</strong></p>
      <ul>
        <li>Endrita Vllasaliu</li>
        <li>Fleta Mujaj</li>
        <li>Rajmondë Shllaku</li>
      </ul>
    </td>
  </tr>
</table>

---

## Përmbajtja

- [Pasqyra e Projektit](#pasqyra-e-projektit)
- [Struktura e Repozitorit](#struktura-e-repozitorit)
- [Përshkrimi i Datasetit](#përshkrimi-i-datasetit)
- [Modulet dhe Skriptat e Implementuara](#modulet-dhe-skriptat-e-implementuara)
- [Analiza Vizuale (EDA)](#analiza-vizuale-eda)
- [Teknologjitë e Përdorura](#teknologjitë-e-përdorura)
- [Instalimi & Konfigurimi](#instalimi--konfigurimi)

---
## Pasqyra e Projektit

Ky repozitor implementon një tubacion (pipeline) gjithëpërfshirës të Machine Learning për parashikimin e **Cilësisë së Ajrit (nivelet e PM2.5 dhe PM10) në Kosovë**, me fokus në analizën krahasuese mes tri qyteteve kryesore: **Prishtinë, Prizren dhe Pejë**. 
Projekti demonstron një cikël të plotë jetësor (end-to-end) të shkencës së të dhënave, duke u shtrirë në tri faza kryesore:

1. **Faza 1 (Përgatitja e të Dhënave):** Mbledhja e të dhënave historike përmes API-ve për të tria qytetet, pastrimi inteligjent i anomalive (Metoda Hibride e bazuar në Njohuritë e Domenit), integrimi i serive kohore, inxhinieria e veçorive dhe krijimi i një **Dataseti Global** të unifikuar.
2. **Faza 2 (Trajnimi i Modelit):** Aplikimi dhe optimizimi i algoritmeve të Machine Learning (si Random Forest, XGBoost, ose Regression) për të gjetur lidhjet e fshehura mes kushteve meteorologjike, lokacionit gjeografik dhe ndotjes së ajrit.
3. **Faza 3 (Analiza dhe Vlerësimi):** Testimi i saktësisë së modeleve, nxjerrja e metrikave dhe krijimi i raporteve vizuale për të kuptuar se cilët faktorë ndikojnë më shumë në smog-un sipas rajoneve.

### Qëllimet e Projektit
- **Ndërtimi i një Modeli Parashikues Rajonal:** Krijimi i një algoritmi të aftë të parashikojë ndotjen bazuar në motin dhe lokacionin (qytetin) për 6 vitet e fundit.
- **Paraprocesimi Inovativ:** Aplikimi i teknikave të avancuara të pastrimit për të menaxhuar dështimet e sensorëve, duke ruajtur vlerat reale të ndotjes ekstreme dimërore.
- **Inxhinieria e Veçorive Globale:** Transformimi i të dhënave në inpute parashikuese numerike (kodimi i qyteteve, nxjerrja e orës/muajit) dhe krijimi i veçorive të domenit si `sezoni_i_ngrohjes` për të kapur "Efektin e Prishtinës".
- **Krahasimi i Algoritmeve:** Trajnimi i modeleve për të gjetur performancën më të lartë dhe gabimin më të vogël.

---

## Struktura e Repozitorit

```text
MachineLearning_25-26_GR-5/
│
├── Data_Gathering/                  
│   └── data_gathering.py            # Skripta e automatizuar për marrjen e të dhënave
│
├── Datasetet/                       
│   ├── unprocessed_datasets/        # Të dhënat e papërpunuara
│   ├── processed_dataset/           # Rezultatet e integrimit fillestar
│   ├── cleaned_dataset/             # Rezultatet pas fshirjes së nulls/outliers
│   └── ml_ready_dataset/            # Datasetet finale numerike dhe ai GLOBAL
│       ├── prishtine_ml_data.csv
│       ├── prizren_ml_data.csv
│       ├── peje_ml_data.csv
│       └── kosova_global_ml_data.csv # DATASETI PER TRAJNIM
│
├── FAZA-1_Pergatitja_e_modelit/     # Faza kryesore e Paraprocesimit
│   ├── data_integration.py          
│   ├── data_cleaning.py             
│   ├── feature_engineering.py       
│   ├── data_merging.py       
│   └── eda_analysis.py              
│
├── FAZA-2_Trajnimi_i_modelit/       # [Në zhvillim] 
│
├── FAZA-3_Analiza_dhe_Evaluimi/     # [Në zhvillim] 
│
├── images/                          # Imazhet e gjeneruara nga EDA (Kombinim Global & Lokal)
├── .gitattributes
├── LICENSE                          # MIT LICENSE
├── README.md                        # Dokumentimi i projektit
└── .gitignore                       
````

-----

## Përshkrimi i Datasetit

Projekti përdor burime të të dhënave që mbulojnë periudhën nga **2020 deri më sot**. Të dhënat e ndara fillimisht janë bashkuar në një skedar master: **`kosova_global_ml_data.csv`**.

### Atributet Kryesore (Dataseti Global)

| Atributi                | Kolona | Tipi | Përshkrimi |
|-------------------------|--------|------|-------------|
| **Targeti (Ajri)** | `pm2_5` | Float | Përqendrimi i grimcave \< 2.5 µm (µg/m³) |
| **Targeti (Ajri)** | `pm10` | Float | Përqendrimi i grimcave \< 10 µm (µg/m³) |
| **Meteorologjike** | `temperature_2m` | Float | Temperatura e ajrit në 2 metra lartësi (°C) |
| **Meteorologjike** | `relative_humidity_2m`| Float | Lagështia relative (%) |
| **Meteorologjike** | `wind_speed_10m` | Float | Shpejtësia e erës në 10 metra lartësi (km/h) |
| **Kohore (E derivuar)** | `month` | Integer | Muaji i vitit (1-12) për të kapur sezonalitetin |
| **Kohore (E derivuar)** | `hour` | Integer | Ora e ditës (0-23) për të kapur ciklet e trafikut |
| **Domeni (E derivuar)** | `sezoni_i_ngrohjes` | Integer | Indikator i smogut (1=Dimër në Prishtinë, 0=Tjera) |
| **Gjeografike** | `qyteti` | Integer | Kodi i lokacionit (**1**=Prishtinë, **2**=Prizren, **3**=Pejë) |

-----

## Modulet dhe Skriptat e Implementuara

### FAZA 1 : Përgatitja e modelit

Kjo fazë zbaton një rrjedhë të fuqishme të paraprocesimit të të dhënave përmes disa skriptave kryesore:

1.  **`data_integration.py`**

      * **Çfarë bën:** Lexon datasetet e papërpunuara të motit dhe cilësisë së ajrit për secilin qytet dhe i bashkon ato në një skedar analitik të vetëm.
      * **Logjika:**  
         * **Sinkronizimi Kohor**: Për të garantuar përputhjen e saktë mes API-ve të ndryshme, skripta konverton formatet e datave, rrumbullakos kohën në orën më të afërt (round('h')) dhe standardizon zonat kohore. 
         * **Pivotimi i të Dhënave**: Shumë burime të ajrit (si OpenAQ) vijnë në format të gjatë (parametrat janë rreshta). Skripta filtron vlerat e korruptuara si -999.0, fshin matjet e dyfishta për të njëjtën orë, dhe përdor funksionin pivot për t'i kthyer ndotësit (PM2.5, PM10) në kolona të rregullta. Nëse detekton format satelitor, e përshtat automatikisht pa pivot. 
         * Në fund, kryhet bashkimi inner në kolonën time, duke ruajtur vetëm rreshtat ku plotësohen të dyja kushtet (moti dhe ajri).


2.  **`data_cleaning.py`**

      * **Çfarë bën:** Së pari, siguron që çdo vlerë tekstuale e gabuar (si 'None') të konvertohet në NaN dhe fshin plotësisht rreshtat që kanë vlera të munguara. Më pas, bën një skanim statistikor për të detektuar anomali (Outliers) në çdo qytet.
      * **Logjika "Hands Off" (Mos-ndërhyrje):** Për të zbuluar vlerat jashtë normales, skripta përdor metodën e rreptë statistikore IQR (Interquartile Range). Megjithatë, ajo vetëm i raporton (numëron) këto vlera dhe nuk i fshin. Kjo është një zgjedhje inxhinierike e qëllimshme: fshirja e rreshtave me PM2.5 ose PM10 shumë të lartë do të "fshinte" ditët me smog ekstrem gjatë dimrit në Kosovë, duke e lënë modelin të verbër ndaj kulmeve të vërteta të ndotjes.

3.  **`feature_engineering.py` (Inxhinieria e Veçorive)**

    * **Çfarë bën:** Merr të dhënat e pastruara dhe i shndërron ato në formate thjesht numerike, të gatshme për t'u ushqyer në modelet e Machine Learning.
    * **Logjika:** 
         
      * **Ekstraktimi Kohor:** Nga kolona e datës (time), nxjerr kolonat month (për sezonalitetin), hour (për ciklet e ditës/natës) dhe day_of_week. 
      * **Njohuritë e Domenit (Domain Knowledge)**: Shton kolonën is_weekend (1 për fundjavë, 0 për ditë pune) për të kapur ndryshimet në fluksin e trafikut të makinave. Gjithashtu, zbaton një rregull specifik gjeografik: shton kolonën sezoni_i_ngrohjes (muajt e ftohtë 10-3) vetëm për qytetin e Prishtinës, pasi aty ndikimi i djegies së thëngjillit është më agresiv. 
      * **Gati për ML:** Në fund, skripta fshin përfundimisht kolonën tekstuale/datetime time, duke lënë pas vetëm atribute numerike (int dhe float) që kërkohen nga algoritme si Random Forest dhe XGBoost.
4.  **`data_merging.py` (Krijimi i Datasetit Global)**

      * **Çfarë bën:** Merr datasetet e gatshme të të tria qyteteve dhe i shton njëra mbi tjetrën (bashkim vertikal) për të krijuar një *Master Dataset* (`kosova_global_ml_data.csv`).
      * **Logjika:** Shton kolonën e re `qyteti` (Kodet 1, 2, 3) për t'i mundësuar modelit të dallojë lokacionin. Gjithashtu, i jep vlerën `0` sezonit të ngrohjes për Prizrenin dhe Pejën, duke e lënë vlerën `1` vetëm për Prishtinën, për të kapur "Efektin e bllokimit të tymit" tipik të kryeqytetit.

5.  **`eda_analysis.py` (Analiza dhe Vizualizimi)**

      * **Çfarë bën:** Skripta finale automatike që lexon datasetet dhe gjeneron një set grafikësh shkencorë për të analizuar sjelljen e të dhënave, duke i ruajtur ato në folderin `images/`.

-----

## Analiza Vizuale (EDA)

Përmes skriptës `eda_analysis.py`, ne analizuam sjelljen e ndotësve si në nivel lokal (qytet) ashtu edhe atë rajonal.

### 1\. Analiza Lokale

Këtu vëzhgojmë korrelacionin (lidhjen mes motit dhe ndotjes) si dhe trendin mesatar të ndotjes gjatë një dite (00:00 - 23:00) për çdo qytet.

#### Prishtina
| Matrica e Korrelacionit | Cikli Ditor (Trendi Orar) |
|:---:|:---:|
| ![Korrelacioni Prishtinë](images/korrelacioni_prishtine.png) | ![Trendi Prishtinë](images/trendi_orar_prishtine.png) |

#### Prizreni
| Matrica e Korrelacionit | Cikli Ditor (Trendi Orar) |
|:---:|:---:|
| ![Korrelacioni Prizren](images/korrelacioni_prizren.png) | ![Trendi Prizren](images/trendi_orar_prizren.png) |


#### Peja
| Matrica e Korrelacionit | Cikli Ditor (Trendi Orar) |
|:---:|:---:|
| ![Korrelacioni Pejë](images/korrelacioni_peje.png) | ![Trendi Pejë](images/trendi_orar_peje.png) |

**Gjetje nga Analiza Lokale:** \* Të tria qytetet shfaqin formën karakteristike "U" gjatë ditës (ndotje më e ulët gjatë drekës, rritje në mbrëmje).

  * Era (Wind Speed) është pastruesi kryesor natyral, pasi shfaq korrelacion negativ në të tria matricat.
  * Ndërsa Prishtina kapërcen lehtësisht vlerat 20-30 µg/m³ gjatë mbrëmjeve, Prizreni dhe Peja mbajnë një balancë dukshëm më të ulët, falë mungesës së centraleve të mëdha industriale dhe relievit më të favorshëm të ajrimit.

-----

### 2\. Analiza Globale (Krahasimi Rajonal)

Duke përdorur Datasetin Master të unifikuar, ne krijuam një hartëvizuale se si diferencojnë qytetet nga njëri-tjetri.

#### Matrica e Korrelacionit Global
![Matrica e Korrelacionit Global](images/korrelacioni_global.png)
*Figura 1.**Matrica e Korrelacionit (Pearson)** për datasetin e unifikuar (`kosova_global_ml_data.csv`).*

#### Krahasimi i Qyteteve (Boxplot)
![Krahasimi i Qyteteve](images/krahasimi_qyteteve_boxplot.png)
*Figura 2: Analiza e variancës së PM2.5 mes Prishtinës, Prizrenit dhe Pejës.*

  * **Ndotja Ekstreme:** Prishtina shfaq një shpërndarje (box) më të gjerë dhe vlera ekstreme (outliers) dukshëm më të larta, duke vërtetuar se smogu i rëndë është kryesisht problem i kryeqytetit.

#### Cikli Ditor i Ndotjes - Krahasim
![Line Chart Krahasues](images/trendi_orar_global.png)
*Figura 3: Trendi mesatar orar (24h) për PM2.5.*

  * **Dinamika Dite-Natë:** Të tria qytetet ndjekin një model të ngjashëm gjatë ditës, por linja e Prishtinës (e kuqe) shkëputet drastikisht pas orës 17:00, duke krijuar atë që quhet "Peak i Mbrëmjes".


### 3. Konkluzione të Avancuara nga EDA (Gjetje Shkencore)
Projekti tani posedon një **Master Dataset** të pastruar nga anomalitë, me mungesa të trajtuara logjikisht, dhe të pasuruar me Features numerike. Jemi gati për t'i ushqyer këto të dhëna në modelet (Random Forest, XGBoost) për të filluar parashikimet\!
Nga vizualizimet e mësipërme, kemi nxjerrë disa përfundime thelbësore që ndikojnë drejtpërdrejt në qasjen tonë të Machine Learning:

* **Inversioni Termik dhe Meteorologjia:** Matrica e korrelacionit tregon një lidhje të fortë negative mes temperaturës dhe ndotjes. Kjo vërteton se smogu në Kosovë nuk është një problem konstant industrial, por një problem sezonal. Temperaturat e ulëta të kombinuara me lagështinë e lartë krijojnë "inversionin termik", duke e mbajtur ndotjen të bllokuar afër sipërfaqes së tokës.
* **Burimi i "Peak-ut" të Mbrëmjes:** Rritja e ndotjes pas orës 17:00 nuk i atribuohet vetëm trafikut. Fakti që nivelet e PM2.5/PM10 vazhdojnë të rriten në mënyrë agresive deri në mesnatë (kur makinat ndalojnë së lëvizuri) tregon se burimi kryesor është **ngrohja shtëpiake** (djegia e biomasës dhe thëngjillit gjatë natës).
* **Ndikimi Topografik (Pse Prishtina vuan më shumë):** *Boxplot*-et tregojnë anomali ekstreme për Prishtinën. Kryeqyteti ka një reliev që favorizon bllokimin e tymrave, ndërsa Peja dhe Prizreni, të pozicionuara pranë masiveve malore (Bjeshkët e Nemuna, Malet e Sharrit), përfitojnë nga rrymat ajrore që ndihmojnë në ventilimin natyror gjatë natës.
* **Justifikimi i Algoritmeve (Natyra Jo-Lineare):** Lidhjet mes motit, lokacionit dhe ndotjes janë thellësisht jo-lineare. Për të kapur këtë kompleksitet, modelet e thjeshta si *Linear Regression* janë të pamjaftueshme. Kjo është arsyeja pse në Fazën 2 përdorim **modele të bazuara në pemë** (*Random Forest*, *XGBoost*), të cilat janë të afta të zbulojnë rregulla komplekse logjike nga të dhënat:
  > *Shembull se si "mendon" modeli:* **NËSE** Ora > 18:00 **DHE** Temperatura < 0°C **DHE** Shpejtësia e Erës < 5 km/h **ATËHERË** Parashiko Ndotje Ekstreme (PM2.5 > 150).
-----

## FAZA 2 : Trajnimi i Modelit [Në Zhvillim]

## FAZA 3 : Analiza dhe Vlerësimi [Në Zhvillim]

-----

## Teknologjitë e Përdorura

  - **Python 3.x** - Gjuha kryesore e programimit
  - **pandas / numpy** - Manipulimi i të dhënave dhe integrimi i serive kohore
  - **matplotlib / seaborn** - Vizualizimi shkencor i të dhënave (EDA)
  - **requests** - Për marrjen e të dhënave përmes API

-----

## Instalimi & Konfigurimi

**1. Klononi repozitorin:**

```bash
 git clone [https://github.com/rajmondashllaku/MachineLearning_25-26_GR-5.git](https://github.com/rajmondashllaku/MachineLearning_25-26_GR-5.git)
 cd MachineLearning_25-26_GR-5
```

**2. Krijoni dhe aktivizoni një ambient virtual:**

```bash
python -m venv .venv
# Aktivizimi (Windows)
.venv\Scripts\activate
# Aktivizimi (macOS/Linux)
source .venv/bin/activate
```

**3. Instaloni libraritë e nevojshme:**

```bash
pip install pandas numpy matplotlib seaborn requests
```

## Licenca

Ky projekt është i licencuar nën kushtet e **MIT License**. Për më shumë detaje, shikoni skedarin [LICENSE](https://www.google.com/search?q=LICENSE) në këtë repozitor.


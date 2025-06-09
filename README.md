
### 💊Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking
This repository contains the implementation details of SCP and POSD to assess the vulnerability of current large models to content generation. At the same time, in order to better show the effect of our work, the complete experimental results of some model sites are provided in the repository.(Note: The current code structure is slightly messy, the code will be adjusted and updated in the future, so as to facilitate the reproduction of subsequent research.)
#### DTD
We first find a phenomenon that as the large model generates more and more content, it is more vulnerable to malicious attacks to jailbreak. Figure 1 shows the change of attention to the head and tail of the cue word as more and more content is generated. Figure 2 shows the uneven degree of model attention on the generated content, that is, the GINI coefficient exceeds, The more the model focuses on the tail of the output.
![alt text](image.png)
![alt text](image-1.png)
- **Motivation:** Therefore, we propose a jailbreak attack paradigm based on large model generation, which makes the large model output Benign content related to malicious problems through the header hint word (Benign), and then uses adversarial reasoning hint words to generate malicious content related to malicious problems.
#### SCP
![alt text](image-2.png)
#### Structure
```  
├── words_negative/       # SCP attack implementation details  
├── defense/            # POSD defense implementation + experimental results  
├── config/             # Unified model and dataset configurations  
└── docs/               # Experimental result reports (to be added)  
```  

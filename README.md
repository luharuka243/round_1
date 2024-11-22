# CyberGuard_IndiaAI_Hackathon

## Use Python>=3.9

- **`main_category`**: High-level crime categories, such as ["Women/Child Related Crime", "Financial Fraud Crimes", "Other Cyber Crime"]  
- **`category`**: Specific crime types within each `main_category`, such as ['Any Other Cyber Crime',
 'Child Pornography CPChild Sexual Abuse Material CSAM',
 'Crime Against Women & Children',
 'Cryptocurrency Crime',
 'Cyber Attack/ Dependent Crimes',
 'Cyber Terrorism',
 'Hacking  Damage to computercomputer system etc',
 'Online Cyber Trafficking',
 'Online Financial Fraud',
 'Online Gambling  Betting',
 'Online and Social Media Related Crime',
 'Ransomware',
 'RapeGang Rape RGRSexually Abusive Content',
 'Report Unlawful Content',
 'Sexually Explicit Act',
 'Sexually Obscene material']
  
- **`sub_category`**: More granular details describing specific offenses under each `category`, such as ['cyber bullying  stalking  sexting', 'fraud callvishing',
       'online gambling  betting', 'online job fraud',
       'upi related frauds', 'internet banking related fraud',
       'rape/gang rape-sexually abusive content',
       'profile hacking identity theft',
       'debitcredit card fraudsim swap fraud', 'ewallet related fraud',
       'data breach/theft',
       'denial of service (dos)/distributed denial of service (ddos) attacks',
       'fakeimpersonating profile', 'cryptocurrency fraud',
       'sale publishing and transmitting obscene material/sexually explicit material',
       'malware attack', 'business email compromiseemail takeover',
       'email hacking', 'cheating by impersonation', 'defacement/hacking',
       'unauthorised accessdata breach', 'sql injection',
       'provocative speech for unlawful acts', 'ransomware attack',
       'cyber terrorism',
       'child pornography/child sexual abuse material (csam)',
       'tampering with computer source documents',
       'dematdepository fraud', 'online trafficking',
       'online matrimonial fraud', 'website defacementhacking',
       'damage to computer computer systems etc', 'impersonating email',
       'email phishing', 'ransomware', 'intimidating email',
       'against interest of sovereignty or integrity of india', 'other',
       'malicious mobile app attacks',
       'aadhar enabled payment system (aeps) fraud',
       'cyber blackmailing/threatening',
       'attacks on applications (e.g., e-governance, e-commerce)',
       'attacks or incidents affecting digital payment systems',
       'fake mobile apps', 'unauthorized social media access',
       'attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics',
       'password attacks', 'disinformation or misinformation campaigns',
       'sexual harassment', 'web application vulnerabilities',
       'compromise of critical systems/information',
       'supply chain attacks',
       'attacks on servers (database mail dns) and network devices (routers)',
       'attacks or suspicious activities affecting cloud computing systems servers software and applications',
       'cyber espionage',
       'attacks on systems related to artificial intelligence (ai) and machine learning (ml)',
       'targeted scanning/probing of critical networks/systems']  
- **`content processed`**: The actual text data used to train models for classification.
### Mapping:
To establish relationships between these columns, we have two key mappings:
- **`main_category_to_category_mapping`**: Maps each `main_category` to its corresponding `category` values.  
- **`category_to_sub_category_mapping`**: Maps each `category` to its respective `sub_category` values.
1. main_category_to_category_mapping = {
  "Women/Child Related Crime": [
    "Child Pornography CPChild Sexual Abuse Material CSAM",
    "Crime Against Women & Children",
    "Online Cyber Trafficking",
    "RapeGang Rape RGRSexually Abusive Content",
    "Sexually Explicit Act",
    "Sexually Obscene material"
  ],
  "Financial Fraud Crimes": [
    "Cryptocurrency Crime",
    "Online Financial Fraud",
    "Online Gambling  Betting"
  ],
  "Other Cyber Crime": [
    "Any Other Cyber Crime",
    "Cyber Attack/ Dependent Crimes",
    "Cyber Terrorism",
    "Hacking  Damage to computercomputer system etc",
    "Online and Social Media Related Crime",
    "Ransomware",
    "Report Unlawful Content"
  ]
}
2. category_to_sub_category_mapping = {
    "online gambling  betting": [
        "online gambling  betting"
    ],
    "cryptocurrency crime": [
        "cryptocurrency fraud"
    ],
    "online cyber trafficking": [
        "online trafficking"
    ],
    "ransomware": [
        "ransomware"
    ],
    "child pornography cpchild sexual abuse material csam": [
        "child pornography/child sexual abuse material (csam)"
    ],
    "sexually explicit act": [
        "sale publishing and transmitting obscene material/sexually explicit material"
    ],
    "hacking  damage to computercomputer system etc": [
        "unauthorised accessdata breach",
        "email hacking",
        "tampering with computer source documents",
        "website defacementhacking",
        "damage to computer computer systems etc"
    ],
    "online financial fraud": [
        "dematdepository fraud",
        "fraud callvishing",
        "internet banking related fraud",
        "business email compromiseemail takeover",
        "upi related frauds",
        "debitcredit card fraudsim swap fraud",
        "ewallet related fraud"
    ],
    "cyber terrorism": [
        "cyber terrorism"
    ],
    "any other cyber crime": [
        "identity theft, spoofing, and phishing attacks ",
        "zero-day exploits",
        "attacks on servers (database mail dns) and network devices (routers)",
        "attacks on applications (e.g., e-governance, e-commerce)",
        "other",
        "attacks on critical infrastructure, scada, operational technology systems, and wireless networks",
        "fake mobile apps",
        "password attacks",
        "attacks or suspicious activities affecting cloud computing systems, servers, software, and applications",
        "disinformation or misinformation campaigns",
        "unauthorized social media access",
        "attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics",
        "attacks or incidents affecting digital payment systems",
        "child pornography/child sexual abuse material (csam)",
        "malicious code attacks (specifically mentioning virus, worm, trojan, bots, spyware, cryptominers)",
        "aadhar enabled payment system (aeps) fraud",
        "attacks or suspicious activities affecting cloud computing systems servers software and applications",
        "cyber blackmailing/threatening",
        "compromise of critical systems/information",
        "supply chain attacks",
        "sale publishing and transmitting obscene material/sexually explicit material",
        "attacks on internet of things (iot) devices and associated systems, networks, and servers",
        "cyber espionage",
        "sexual harassment",
        "web application vulnerabilities",
        "attacks on systems related to artificial intelligence (ai) and machine learning (ml)",
        "targeted scanning/probing of critical networks/systems",
        "data leaks",
        "malicious mobile app attacks"
    ],
    "crime against women & children": [
        "sexual harassment",
        "computer generated csam/csem",
        "cyber blackmailing & threatening"
    ],
    "rapegang rape rgrsexually abusive content": [
        "rape/gang rape-sexually abusive content"
    ],
    "sexually obscene material": [
        "sale publishing and transmitting obscene material/sexually explicit material"
    ],
    "report unlawful content": [
        "against interest of sovereignty or integrity of india"
    ],
    "cyber attack/ dependent crimes": [
        "hacking/defacement",
        "sql injection",
        "tampering with computer source documents",
        "malware attack",
        "denial of service (dos)/distributed denial of service (ddos) attacks",
        "ransomware attack",
        "data breach/theft",
        "defacement/hacking"
    ],
    "online and social media related crime": [
        "intimidating email",
        "cyber bullying  stalking  sexting",
        "impersonating email",
        "profile hacking identity theft",
        "online job fraud",
        "email phishing",
        "provocative speech for unlawful acts",
        "fakeimpersonating profile",
        "online matrimonial fraud",
        "cheating by impersonation"
    ]
}
### Goal:
To build a series of machine learning models tailored for different levels of classification granularity.
---
### Models to Build:
1. **Model 1**:  
   - **Purpose**: Classify data into one of the three `main_category` values:  
     ["Women/Child Related Crime", "Financial Fraud Crimes", "Other Cyber Crime"]
2. **Model 2**:  
   - **Purpose**: Classify data into `category` values under **"Women/Child Related Crime."**  
   - **Classes**:  
     [
    "Child Pornography CPChild Sexual Abuse Material CSAM",
    "Crime Against Women & Children",
    "Online Cyber Trafficking",
    "RapeGang Rape RGRSexually Abusive Content",
    "Sexually Explicit Act",
    "Sexually Obscene material"
  ]
3. **Model 3**:  
   - **Purpose**: Classify data into `category` values under **"Financial Fraud Crimes."**  
   - **Classes**:  
     [
    "Cryptocurrency Crime",
    "Online Financial Fraud",
    "Online Gambling  Betting"
  ]
4. **Model 4**:  
   - **Purpose**: Classify data into `category` values under **"Other Cyber Crime."**  
   - **Classes**:  
     [
    "Any Other Cyber Crime",
    "Cyber Attack/ Dependent Crimes",
    "Cyber Terrorism",
    "Hacking  Damage to computercomputer system etc",
    "Online and Social Media Related Crime",
    "Ransomware",
    "Report Unlawful Content"
  ]  
5. **Model 5**:  
   - **Purpose**: Classify data into `sub_category` values under **"Hacking/Damage to computer system, etc."**  
   - **Classes**:  
     [
        "unauthorised accessdata breach",
        "email hacking",
        "tampering with computer source documents",
        "website defacementhacking",
        "damage to computer computer systems etc"
    ] 
6. **Model 6**:  
   - **Purpose**: Classify data into `sub_category` values under **"Online Financial Fraud."**  
   - **Classes**:  
     [
        "dematdepository fraud",
        "fraud callvishing",
        "internet banking related fraud",
        "business email compromiseemail takeover",
        "upi related frauds",
        "debitcredit card fraudsim swap fraud",
        "ewallet related fraud"
    ]  
7. **Model 7**:  
   - **Purpose**: Classify data into `sub_category` values under **"Any Other Cyber Crime."**  
   - **Classes**:  
     Includes a range of miscellaneous offenses such as:  
     [
        "identity theft, spoofing, and phishing attacks ",
        "zero-day exploits",
        "attacks on servers (database mail dns) and network devices (routers)",
        "attacks on applications (e.g., e-governance, e-commerce)",
        "other",
        "attacks on critical infrastructure, scada, operational technology systems, and wireless networks",
        "fake mobile apps",
        "password attacks",
        "attacks or suspicious activities affecting cloud computing systems, servers, software, and applications",
        "disinformation or misinformation campaigns",
        "unauthorized social media access",
        "attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics",
        "attacks or incidents affecting digital payment systems",
        "child pornography/child sexual abuse material (csam)",
        "malicious code attacks (specifically mentioning virus, worm, trojan, bots, spyware, cryptominers)",
        "aadhar enabled payment system (aeps) fraud",
        "attacks or suspicious activities affecting cloud computing systems servers software and applications",
        "cyber blackmailing/threatening",
        "compromise of critical systems/information",
        "supply chain attacks",
        "sale publishing and transmitting obscene material/sexually explicit material",
        "attacks on internet of things (iot) devices and associated systems, networks, and servers",
        "cyber espionage",
        "sexual harassment",
        "web application vulnerabilities",
        "attacks on systems related to artificial intelligence (ai) and machine learning (ml)",
        "targeted scanning/probing of critical networks/systems",
        "data leaks",
        "malicious mobile app attacks"
    ]
8. **Model 8**:  
   - **Purpose**: Classify data into `sub_category` values under **"Crime Against Women & Children."**  
   - **Classes**:  
     [
        "sexual harassment",
        "computer generated csam/csem",
        "cyber blackmailing & threatening"
    ]
9. **Model 9**:  
   - **Purpose**: Classify data into `sub_category` values under **"Cyber Attack/Dependent Crimes."**  
   - **Classes**:  
     [
        "hacking/defacement",
        "sql injection",
        "tampering with computer source documents",
        "malware attack",
        "denial of service (dos)/distributed denial of service (ddos) attacks",
        "ransomware attack",
        "data breach/theft",
        "defacement/hacking"
    ]
10. **Model 10**:  
    - **Purpose**: Classify data into `sub_category` values under **"Online and Social Media Related Crime."**  
    - **Classes**:  
      [
        "intimidating email",
        "cyber bullying  stalking  sexting",
        "impersonating email",
        "profile hacking identity theft",
        "online job fraud",
        "email phishing",
        "provocative speech for unlawful acts",
        "fakeimpersonating profile",
        "online matrimonial fraud",
        "cheating by impersonation"
    ]


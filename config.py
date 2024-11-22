# Mappings
category_names_to_category ={
            "women/child related crime": [
                "child pornography cpchild sexual abuse material csam",
                "crime against women & children",
                "online cyber trafficking",
                "rapegang rape rgrsexually abusive content",
                "sexually explicit act",
                "sexually obscene material"
            ],
            "financial fraud crimes": [
                "cryptocurrency crime",
                "online financial fraud",
                "online gambling  betting"
            ],
            "other cyber crime": [
                "any other cyber crime",
                "cyber attack/ dependent crimes",
                "cyber terrorism",
                "hacking  damage to computercomputer system etc",
                "online and social media related crime",
                "report unlawful content"
            ]
        }

category_to_sub_category ={
            "any other cyber crime": [
                "other",
                "supply chain attacks"
            ],
            "child pornography cpchild sexual abuse material csam": [
                "child pornography cpchild sexual abuse material csam"
            ],
            "crime against women & children": [
                "sexual harassment",
                "computer generated csam/csem"
            ],
            "cryptocurrency crime": [
                "cryptocurrency fraud"
            ],
            "cyber attack/ dependent crimes": [
                "sql injection",
                "ransomware attack",
                "malware attack",
                "malicious code attacks (specifically mentioning virus, worm, trojan, bots, spyware, cryptominers)",
                "data breach/theft",
                "data leaks",
                "hacking/defacement",
                "zero-day exploits",
                "malicious mobile app attacks",
                "denial of service (dos)/distributed denial of service (ddos) attacks",
                "tampering with computer source documents"
            ],
            "cyber terrorism": [
                "cyber terrorism",
                "cyber espionage"
            ],
            "hacking  damage to computercomputer system etc": [
                "email hacking",
                "unauthorised accessdata breach",
                "compromise of critical systems/information",
                "targeted scanning/probing of critical networks/systems",
                "attacks on servers (database mail dns) and network devices (routers)",
                "attacks on critical infrastructure, scada, operational technology systems, and wireless networks",
                "attacks or suspicious activities affecting cloud computing systems servers software and applications",
                "attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics",
                "attacks on internet of things (iot) devices and associated systems, networks, and servers",
                "attacks on systems related to artificial intelligence (ai) and machine learning (ml)",
                "damage to computer computer systems etc",
                "web application vulnerabilities",
            ],
            "online cyber trafficking": [
                "online trafficking"
            ],
            "online financial fraud": [
                "upi related frauds",
                "aadhar enabled payment system (aeps) fraud",
                "business email compromiseemail takeover",
                "debitcredit card fraudsim swap fraud",
                "ewallet related fraud",
                "fraud callvishing",
                "internet banking related fraud",
                "attacks or incidents affecting digital payment systems"
            ],
            "online gambling  betting": [
                "online gambling  betting"
            ],
            "online and social media related crime": [
                "intimidating email",
                "provocative speech for unlawful acts",
                "email phishing",
                "online job fraud",
                "profile hacking identity theft",
                "identity theft, spoofing, and phishing attacks",
                "unauthorized social media access",
                "cheating by impersonation",
                "fake mobile apps",
                "online matrimonial fraud",
                "cyber bullying  stalking  sexting",
                "fakeimpersonating profile"
            ],
            "rapegang rape rgrsexually abusive content": [
                "rapegang rape rgrsexually abusive content"
            ],
            "report unlawful content": [
                "against interest of sovereignty or integrity of india",
                "disinformation or misinformation campaigns"
            ],
            "sexually explicit act": [
                "sexually explicit act"
            ],
            "sexually obscene material": [
                "sale publishing and transmitting obscene material/sexually explicit material"
            ]
        }


master_mapper = {
    "any other cyber crime": {
        "other": [
            "other",
            "supply chain attacks"
        ]
    },
    "child pornography cpchild sexual abuse material csam": {
        "child pornography cpchild sexual abuse material csam": [
            "child pornography cpchild sexual abuse material csam"
        ]
    },
    "crime against women & children": {
        "sexual harassment": [
            "sexual harassment"
        ],
        "computer generated csam/csem": [
            "computer generated csam/csem"
        ]
    },
    "cryptocurrency crime": {
        "cryptocurrency fraud": [
            "cryptocurrency fraud"
        ]
    },
    "cyber attack/ dependent crimes": {
        "sql injection": [
            "sql injection"
        ],
        "ransomware attack": [
            "ransomware attack"
        ],
        "malware attack": [
            "malware attack",
            "malicious code attacks (specifically mentioning virus, worm, trojan, bots, spyware, cryptominers)"
        ],
        "data breach/theft": [
            "data breach/theft",
            "data leaks"
        ],
        "hacking/defacement": [
            "hacking/defacement",
            "zero-day exploits",
            "malicious mobile app attacks"
        ],
        "denial of service (dos)/distributed denial of service (ddos) attacks": [
            "denial of service (dos)/distributed denial of service (ddos) attacks"
        ],
        "tampering with computer source documents": [
            "tampering with computer source documents"
        ]
    },
    "cyber terrorism": {
        "cyber terrorism": [
            "cyber terrorism",
            "cyber espionage"
        ]
    },
    "hacking  damage to computercomputer system etc": {
        "email hacking": [
            "email hacking"
        ],
        "unauthorised accessdata breach": [
            "unauthorised accessdata breach",
            "compromise of critical systems/information",
            "targeted scanning/probing of critical networks/systems",
            "attacks on servers (database mail dns) and network devices (routers)",
            "attacks on critical infrastructure, scada, operational technology systems, and wireless networks",
            "attacks or suspicious activities affecting cloud computing systems servers software and applications",
            "attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics",
            "attacks on internet of things (iot) devices and associated systems, networks, and servers",
            "attacks on systems related to artificial intelligence (ai) and machine learning (ml)"
        ],
        "damage to computer computer systems etc": [
            "damage to computer computer systems etc"
        ],
        "website defacementhacking": [
            "web application vulnerabilities",
        ]
    },
    "online cyber trafficking": {
        "online trafficking": [
            "online trafficking"
        ]
    },
    "online financial fraud": {
        "upi related frauds": [
            "upi related frauds",
            "aadhar enabled payment system (aeps) fraud"
        ],
        "business email compromiseemail takeover": [
            "business email compromiseemail takeover"
        ],
        "debitcredit card fraudsim swap fraud": [
            "debitcredit card fraudsim swap fraud"
        ],
        "ewallet related fraud": [
            "ewallet related fraud"
        ],
        "fraud callvishing": [
            "fraud callvishing"
        ],
        "internet banking related fraud": [
            "internet banking related fraud",
            "attacks or incidents affecting digital payment systems"
        ]
    },
    "online gambling  betting": {
        "online gambling  betting": [
            "online gambling  betting"
        ]
    },
    "online and social media related crime": {
        "intimidating email": [
            "intimidating email"
        ],
        "provocative speech for unlawful acts": [
            "provocative speech for unlawful acts"
        ],
        "email phishing": [
            "email phishing"
        ],
        "online job fraud": [
            "online job fraud"
        ],
        "profile hacking identity theft": [
            "profile hacking identity theft",
            "identity theft, spoofing, and phishing attacks",
            "unauthorized social media access"
        ],
        "cheating by impersonation": [
            "cheating by impersonation",
            "fake mobile apps"
        ],
        "online matrimonial fraud": [
            "online matrimonial fraud"
        ],
        "cyber bullying  stalking  sexting": [
            "cyber bullying  stalking  sexting"
        ],
        "fakeimpersonating profile": [
            "fakeimpersonating profile"
        ]
    },
    "rapegang rape rgrsexually abusive content": {
        "rapegang rape rgrsexually abusive content": [
            "rapegang rape rgrsexually abusive content"
        ]
    },
    "report unlawful content": {
        "against interest of sovereignty or integrity of india": [
            "against interest of sovereignty or integrity of india",
            "disinformation or misinformation campaigns"
        ]
    },
    "sexually explicit act": {
        "sexually explicit act": [
            "sexually explicit act"
        ]
    },
    "sexually obscene material": {
        "sexually obscene material": [
            "sale publishing and transmitting obscene material/sexually explicit material",
            "sexually obscene material"
        ]
    }
}
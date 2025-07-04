MODEL_CONFIG = {
    "weight":{
        "protein": "poc_pre_220816.pt",
        "molecule_no_h": "mol_pre_no_h_220816.pt",
        "molecule_all_h": "pretrain_Tg_0.pth", #exchange the weight
        "crystal": "mp_all_h_230313.pt",
        "mof": "mof_pre_no_h_CORE_MAP_20230505.pt",
        "oled": "oled_pre_no_h_230101.pt",
    },
    "dict":{
        "protein": "poc.dict.txt",
        "molecule_no_h": "mol.dict.txt",
        "molecule_all_h": "mol.dict.txt",
        "crystal": "mp.dict.txt",
        "mof": "mof.dict.txt",
        "oled": "oled.dict.txt",
    },
}
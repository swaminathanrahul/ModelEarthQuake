feature_set = {
    'Basic_Structural': [
        'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage'
    ],
    'Construction_Materials': [
        'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type'
    ],
    'Superstructure_Materials': [
        'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
        'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
        'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
        'has_superstructure_timber', 'has_superstructure_bamboo',
        'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
        'has_superstructure_other'
    ],
    'Usage_and_Legal': [
        'legal_ownership_status', 'count_families', 'has_secondary_use',
        'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental',
        'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
        'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
        'has_secondary_use_other'
    ]
}


def get_features(*categories):
    return [feature for category in categories for feature in feature_set.get(category, [])]

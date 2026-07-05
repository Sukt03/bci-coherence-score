from bci_repro.scoring import aggregate_scores, parse_label_reason_response


def test_parse_label_reason_response_from_json():
    answer, reason = parse_label_reason_response('{"answer":"somewhat","reasoning":"Shape is partially preserved."}')
    assert answer == "somewhat"
    assert reason == "Shape is partially preserved."


def test_aggregate_object_scores():
    response = {
        "routing": "object",
        "perceptual": {
            "P1_global_spatial_structure": {"answer": "yes"},
            "P2_object_shape_silhouette": {"answer": "somewhat"},
            "P3_surface_texture_material": {"answer": "no"},
            "P4_color_chromatic_consistency": {"answer": "yes"},
            "P5_artifact_absence": {"answer": "yes"},
            "P6_holistic_visual_recoverability": {"answer": "no"},
        },
        "semantic": {
            "S1_basic_category_identity": {"answer": "yes"},
            "S2_subordinate_identity": {"answer": "no"},
            "S3_functional_role_purpose": {"answer": "somewhat"},
            "S4_quantity_cardinality": {"answer": "yes"},
            "S5_scene_context_environment": {"answer": "no"},
            "S6_semantic_recoverability": {"answer": "no"},
        },
    }
    scores = aggregate_scores(response)
    assert scores["T_PAS"] == (1 + 0.5 + 0 + 1 + 1 + 0) / 6
    assert scores["T_SAS"] == (1 + 0 + 0.5 + 1 + 0 + 0) / 6


def test_aggregate_abstract_has_no_semantic_score():
    response = {
        "routing": "abstract",
        "perceptual": {
            "P1_global_spatial_structure": "yes",
            "P3_surface_texture_pattern": "somewhat",
            "P4_color_chromatic_consistency": "no",
            "P5_artifact_absence": "yes",
            "P6_holistic_visual_recoverability": "no",
        },
    }
    scores = aggregate_scores(response)
    assert scores["T_PAS"] == 0.5
    assert scores["T_SAS"] is None


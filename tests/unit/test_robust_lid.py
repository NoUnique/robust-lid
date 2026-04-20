import pytest

from robust_lid.ensemble import RobustLID
from robust_lid.utils import ISOConverter, set_converter
from tests.fixtures.fake_lid import FakeLID


@pytest.fixture(autouse=True)
def _inject_small_converter(small_converter: ISOConverter) -> None:
    # normalize_language_code inside models uses the global; pre-seed it.
    set_converter(small_converter)


@pytest.mark.unit
def test_predict_with_fake_models_produces_script_suffix() -> None:
    ensemble = RobustLID(
        models=[
            FakeLID([("eng", 0.9)]),
            FakeLID([("eng", 0.7)]),
            FakeLID([("fra", 0.5)]),
        ]
    )
    code, confidence = ensemble.predict("Hello world")
    assert code == "eng_Latn"
    assert confidence == pytest.approx((0.9 + 0.7) / (0.9 + 0.7 + 0.5))


@pytest.mark.unit
def test_predict_all_empty_returns_sentinel() -> None:
    ensemble = RobustLID(models=[FakeLID([]), FakeLID([])])
    assert ensemble.predict("whatever") == ("und_Zyyy", 0.0)


@pytest.mark.unit
def test_predict_passes_text_to_each_model() -> None:
    fakes = [FakeLID([("eng", 1.0)]) for _ in range(3)]
    ensemble = RobustLID(models=list(fakes))
    ensemble.predict("some text")
    for fake in fakes:
        assert fake.calls == ["some text"]


@pytest.mark.unit
def test_predict_korean_text_gets_hang_script() -> None:
    ensemble = RobustLID(models=[FakeLID([("kor", 0.9)])])
    code, _conf = ensemble.predict("안녕하세요")
    assert code == "kor_Hang"


@pytest.mark.unit
def test_default_models_includes_cld3_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from robust_lid import ensemble as ens

    for name in [
        "LangidLID",
        "LangdetectLID",
        "CLD2LID",
        "CLD3LID",
        "FastText176LID",
        "FastText218eLID",
        "GlotLID",
    ]:
        monkeypatch.setattr(ens, name, lambda *_a, **_kw: FakeLID([]))
    monkeypatch.setattr(ens, "is_cld3_available", lambda: True)

    # fast_mode=False surfaces all 7 backends
    assert len(RobustLID(fast_mode=False).models) == 7
    # fast_mode=True (default) drops the 2 pure-Python backends
    assert len(RobustLID().models) == 5


@pytest.mark.unit
def test_default_models_excludes_cld3_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from robust_lid import ensemble as ens

    for name in [
        "LangidLID",
        "LangdetectLID",
        "CLD2LID",
        "CLD3LID",
        "FastText176LID",
        "FastText218eLID",
        "GlotLID",
    ]:
        monkeypatch.setattr(ens, name, lambda *_a, **_kw: FakeLID([]))
    monkeypatch.setattr(ens, "is_cld3_available", lambda: False)

    # fast_mode=False: 7-cld3 = 6 backends
    assert len(RobustLID(fast_mode=False).models) == 6
    # fast_mode=True (default): 6 - 2 slow = 4 backends
    assert len(RobustLID().models) == 4


@pytest.mark.unit
def test_ensemble_still_predicts_without_cld3() -> None:
    """Without CLD3 the 6-model ensemble must still produce a valid result."""
    ensemble = RobustLID(
        models=[
            FakeLID([("eng", 0.9)]),  # langid
            FakeLID([("eng", 0.8)]),  # langdetect
            FakeLID([("eng", 0.85)]),  # cld2
            FakeLID([("eng", 0.7)]),  # ft176
            FakeLID([("eng", 0.95)]),  # ft218e
            FakeLID([("eng", 0.88)]),  # glotlid
        ]
    )
    code, conf = ensemble.predict("Hello world")
    assert code == "eng_Latn"
    assert conf == pytest.approx(1.0)


@pytest.mark.unit
def test_weights_change_decision() -> None:
    # Three langid-ish models vote English; one ft218e-ish model votes French.
    # With trust weights favouring ft218e, French wins.
    ensemble = RobustLID(
        models=[
            FakeLID([("eng", 0.6)]),
            FakeLID([("eng", 0.6)]),
            FakeLID([("eng", 0.6)]),
            FakeLID([("fra", 0.9)]),
        ],
        weights=[0.1, 0.1, 0.1, 10.0],
    )
    code, _conf = ensemble.predict("Bonjour le monde")
    assert code.startswith("fra_")


@pytest.mark.unit
def test_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="weights length"):
        RobustLID(models=[FakeLID([])], weights=[1.0, 1.0])


@pytest.mark.unit
def test_script_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="script_weights length"):
        RobustLID(models=[FakeLID([])], script_weights=[{}, {}])


@pytest.mark.unit
def test_lang_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="lang_weights length"):
        RobustLID(models=[FakeLID([])], lang_weights=[{}, {}])


@pytest.mark.unit
def test_script_weight_silences_weak_backend_on_matching_script() -> None:
    """Two backends agree on 'zho' but the third (with script weight 0.0
    on Hani) is silenced on Chinese text → the two winners dominate."""
    ensemble = RobustLID(
        models=[
            FakeLID([("zho", 0.9)]),
            FakeLID([("zho", 0.9)]),
            FakeLID([("kor", 5.0)]),  # very high confidence but wrong
        ],
        weights=[1.0, 1.0, 1.0],
        script_weights=[{}, {}, {"Hani": 0.0}],
    )
    # Chinese-looking text triggers Hani script
    code, _conf = ensemble.predict("你好")
    assert code == "zho_Hani"


@pytest.mark.unit
def test_lang_weight_downweights_specific_prediction() -> None:
    """A backend's ltz prediction is heavily downweighted."""
    ensemble = RobustLID(
        models=[
            FakeLID([("ltz", 0.99)]),  # will be downweighted to ~0
            FakeLID([("deu", 0.5)]),
        ],
        weights=[1.0, 1.0],
        lang_weights=[{"ltz": 0.0}, {}],
    )
    code, _conf = ensemble.predict("Hallo Welt")
    assert code.startswith("deu_")


@pytest.mark.unit
def test_defaults_auto_applied_when_models_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """RobustLID() without args applies all three default tables (fast_mode=True)."""
    from robust_lid import ensemble as ens

    for name in [
        "LangidLID",
        "LangdetectLID",
        "CLD2LID",
        "CLD3LID",
        "FastText176LID",
        "FastText218eLID",
        "GlotLID",
    ]:
        monkeypatch.setattr(ens, name, lambda *_a, **_kw: FakeLID([]))
    monkeypatch.setattr(ens, "is_cld3_available", lambda: True)

    lid = RobustLID()
    assert lid.weights == ens.default_weights(fast_mode=True)
    assert lid.script_weights == ens.default_script_weights(fast_mode=True)
    assert lid.lang_weights == ens.default_lang_weights(fast_mode=True)


@pytest.mark.unit
def test_defaults_not_applied_for_custom_models() -> None:
    """When the user injects custom models, defaults must NOT be applied
    (name-tuned weights don't make sense for arbitrary backends)."""
    lid = RobustLID(models=[FakeLID([]), FakeLID([])])
    assert lid.weights is None
    assert lid.script_weights is None
    assert lid.lang_weights is None


@pytest.mark.unit
def test_default_weights_ft176_is_boosted() -> None:
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()
    weights = ens.default_weights()
    idx_ft176 = order.index("ft176")
    idx_langid = order.index("langid")
    assert weights[idx_ft176] > weights[idx_langid]


@pytest.mark.unit
def test_default_script_weights_downweight_langdetect_on_hani() -> None:
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()
    sw = ens.default_script_weights()
    idx = order.index("langdetect")
    assert sw[idx].get("Hani", 1.0) < 1.0


@pytest.mark.unit
def test_default_script_weights_downweight_glotlid_on_deva() -> None:
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()
    sw = ens.default_script_weights()
    idx = order.index("glotlid")
    assert sw[idx].get("Deva", 1.0) < 1.0


@pytest.mark.unit
def test_default_script_weights_zero_out_cld2_and_cld3_on_hebrew() -> None:
    """cld2 and cld3 both fail completely (0 % recall) on Hebrew in
    WiLi-2018. The defaults must zero them out on Hebr script."""
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()
    sw = ens.default_script_weights()
    assert sw[order.index("cld2")].get("Hebr", 1.0) == 0.0
    if "cld3" in order:
        assert sw[order.index("cld3")].get("Hebr", 1.0) == 0.0


# --- script gating via supported_scripts ---


class _LimitedLID(FakeLID):
    """FakeLID that only declares support for certain scripts."""

    def __init__(self, response: list[tuple[str, float]], scripts: set[str]) -> None:
        super().__init__(response)
        self._scripts = frozenset(scripts)

    @property
    def supported_scripts(self) -> frozenset[str]:
        return self._scripts


@pytest.mark.unit
def test_script_gating_zeros_unsupported_backend() -> None:
    """A backend that doesn't support the input's script must have its vote
    silenced — even if it returned a high-confidence prediction."""
    supporter = FakeLID([("eng", 0.5)])  # base FakeLID has empty support set → no gating
    bad_guess = _LimitedLID([("kor", 0.99)], scripts={"Latn"})  # claims only Latn
    ensemble = RobustLID(models=[supporter, bad_guess])
    code, _ = ensemble.predict("你好")  # Hani script, not in bad_guess.supported_scripts
    assert code.startswith("eng_")


@pytest.mark.unit
def test_script_gating_honored_backend_still_votes() -> None:
    """When a backend's declared support matches the input script, it votes."""
    supporter = _LimitedLID([("zho", 0.9)], scripts={"Hani"})
    ensemble = RobustLID(models=[supporter])
    code, _ = ensemble.predict("你好")
    assert code.startswith("zho_")


@pytest.mark.unit
def test_script_gating_skipped_for_undefined_script() -> None:
    """When detect_script returns Zyyy (no recognizable script), gating must
    NOT run — otherwise every declared backend would be silenced."""
    supporter = _LimitedLID([("eng", 0.9)], scripts={"Latn"})
    ensemble = RobustLID(models=[supporter])
    code, _ = ensemble.predict("12345!!!")  # only digits/punctuation → Zyyy
    # predict returns sentinel since no meaningful vote path, but importantly
    # the backend was NOT zeroed just because the script is unknown.
    assert code.split("_")[0] == "eng"


@pytest.mark.unit
def test_empty_supported_scripts_means_no_gating() -> None:
    """A backend that declares frozenset() (the default) should never be
    gated out — it's the opt-in signal."""
    undeclared = FakeLID([("kor", 0.99)])  # base class default = frozenset()
    assert undeclared.supported_scripts == frozenset()
    ensemble = RobustLID(models=[undeclared])
    code, _ = ensemble.predict("你好")  # Hani
    # Vote is kept even though 'kor' disagrees with the script — because the
    # backend didn't opt in to gating.
    assert code.split("_")[0] == "kor"


@pytest.mark.unit
def test_default_lang_weights_downweight_langid_ltz_and_kir() -> None:
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()
    lw = ens.default_lang_weights()
    idx = order.index("langid")
    assert lw[idx].get("ltz", 1.0) < 1.0
    assert lw[idx].get("kir", 1.0) < 1.0


@pytest.mark.unit
def test_parallel_default_produces_same_result_as_sequential() -> None:
    models = [FakeLID([("eng", 0.9)]), FakeLID([("eng", 0.7)]), FakeLID([("fra", 0.5)])]
    r_par = RobustLID(models=list(models), parallel=True)
    r_seq = RobustLID(models=list(models), parallel=False)
    assert r_par.predict("Hello world") == r_seq.predict("Hello world")


@pytest.mark.unit
def test_parallel_single_model_falls_back_to_sequential() -> None:
    """With one backend the thread-pool path is skipped (pointless overhead)."""
    r = RobustLID(models=[FakeLID([("eng", 1.0)])], parallel=True)
    assert r.predict("Hello")[0] == "eng_Latn"


@pytest.mark.unit
def test_parallel_each_backend_is_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = [FakeLID([("eng", 1.0)]) for _ in range(4)]
    ensemble = RobustLID(models=list(fakes), parallel=True)
    ensemble.predict("hi")
    for fake in fakes:
        assert fake.calls == ["hi"]


@pytest.mark.unit
def test_low_memory_requires_default_models() -> None:
    with pytest.raises(ValueError, match="low_memory=True"):
        RobustLID(models=[FakeLID([])], low_memory=True)


@pytest.mark.unit
def test_default_factories_tracks_cld3_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robust_lid import ensemble as ens

    monkeypatch.setattr(ens, "is_cld3_available", lambda: True)
    assert len(ens._default_factories()) == 7  # fast_mode=False default for inspection
    assert len(ens._default_factories(fast_mode=True)) == 5
    monkeypatch.setattr(ens, "is_cld3_available", lambda: False)
    assert len(ens._default_factories()) == 6
    assert len(ens._default_factories(fast_mode=True)) == 4


@pytest.mark.unit
def test_low_memory_uses_factories_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    from robust_lid import ensemble as ens

    calls: list[str] = []

    def make_factory(name: str) -> object:
        def _f() -> FakeLID:
            calls.append(f"construct:{name}")
            return FakeLID([(name[:3], 0.9)])

        return _f

    factories = [make_factory(n) for n in ("eng", "kor", "jpn")]
    monkeypatch.setattr(ens, "_default_factories", lambda _fm=False: factories)

    lid = RobustLID(
        low_memory=True,
        # Shrink defaults to 3 entries to match our fake factory count
        weights=[1.0, 1.0, 1.0],
        script_weights=[{}, {}, {}],
        lang_weights=[{}, {}, {}],
    )
    lid.predict("Hello")
    # Each factory invoked exactly once per predict call.
    assert calls == ["construct:eng", "construct:kor", "construct:jpn"]

    # Second call reinstantiates — we never keep live references.
    lid.predict("Hello again")
    assert calls == [
        "construct:eng",
        "construct:kor",
        "construct:jpn",
        "construct:eng",
        "construct:kor",
        "construct:jpn",
    ]


@pytest.mark.unit
def test_low_memory_disables_script_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """In low-memory mode we don't hold live model instances, so the supported-scripts
    gate is skipped. A backend whose predict disagrees with the script still votes."""
    from robust_lid import ensemble as ens

    class _LimitedFake(FakeLID):
        supported_scripts = frozenset({"Latn"})  # claims no Hani support

    factories = [lambda: _LimitedFake([("eng", 0.99)])]
    monkeypatch.setattr(ens, "_default_factories", lambda _fm=False: factories)
    lid = RobustLID(low_memory=True, weights=[1.0], script_weights=[{}], lang_weights=[{}])
    code, _ = lid.predict("你好")  # Hani script → would be gated in fast mode
    assert code.startswith("eng_")  # vote kept → gating disabled


@pytest.mark.unit
def test_fast_mode_is_default_true() -> None:
    """RobustLID() without args must enable fast_mode (the whole point)."""
    lid = RobustLID(models=[FakeLID([])])
    assert lid.fast_mode is True


@pytest.mark.unit
def test_fast_mode_default_true_drops_slow_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default RobustLID() drops langid + langdetect in fast_mode."""
    from robust_lid import ensemble as ens

    order = ens.default_backend_order()  # fast_mode=False (all)
    fast_order = ens.default_backend_order(fast_mode=True)
    assert "langid" in order and "langdetect" in order
    assert "langid" not in fast_order and "langdetect" not in fast_order
    assert len(order) - len(fast_order) == 2


@pytest.mark.unit
def test_fast_mode_false_restores_all_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    from robust_lid import ensemble as ens

    for name in [
        "LangidLID",
        "LangdetectLID",
        "CLD2LID",
        "CLD3LID",
        "FastText176LID",
        "FastText218eLID",
        "GlotLID",
    ]:
        monkeypatch.setattr(ens, name, lambda *_a, **_kw: FakeLID([]))
    monkeypatch.setattr(ens, "is_cld3_available", lambda: True)

    all_backends = RobustLID(fast_mode=False)
    assert len(all_backends.models) == 7
    assert all_backends.fast_mode is False


@pytest.mark.unit
def test_slow_backend_names_constant_identifies_pure_python_backends() -> None:
    from robust_lid import ensemble as ens

    assert frozenset({"langid", "langdetect"}) == ens.SLOW_BACKEND_NAMES


@pytest.mark.unit
def test_default_weights_lengths_follow_fast_mode() -> None:
    from robust_lid import ensemble as ens

    full = ens.default_weights()
    fast = ens.default_weights(fast_mode=True)
    assert len(full) - len(fast) == 2  # dropped 2 slow backends

    # Every fast-mode entry must correspond to a backend whose name is NOT slow
    order = ens.default_backend_order(fast_mode=True)
    assert set(order).isdisjoint(ens.SLOW_BACKEND_NAMES)


@pytest.mark.unit
def test_script_weights_do_not_apply_to_other_scripts() -> None:
    """langdetect downweight on Hani must not affect a Latin-script prediction."""
    from robust_lid import ensemble as ens

    # Build uniform weights + just the langdetect-Hani downweight
    langdetect_idx = ens.default_backend_order().index("langdetect")
    sw: list[dict[str, float]] = [{} for _ in ens.default_backend_order()]
    sw[langdetect_idx] = {"Hani": 0.0}

    models: list = [FakeLID([]) for _ in range(len(sw))]
    models[langdetect_idx] = FakeLID([("eng", 1.0)])

    ensemble = RobustLID(models=models, script_weights=sw)
    code, _ = ensemble.predict("Hello world")  # Latin script
    # langdetect's vote was NOT suppressed (script is Latn), so we see eng.
    assert code.startswith("eng_")


@pytest.mark.unit
def test_default_backend_order_matches_default_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robust_lid import ensemble as ens

    for name in [
        "LangidLID",
        "LangdetectLID",
        "CLD2LID",
        "CLD3LID",
        "FastText176LID",
        "FastText218eLID",
        "GlotLID",
    ]:
        monkeypatch.setattr(ens, name, lambda *_a, **_kw: FakeLID([]))

    monkeypatch.setattr(ens, "is_cld3_available", lambda: True)
    assert len(ens.default_backend_order()) == 7
    assert len(ens._default_models()) == 7

    monkeypatch.setattr(ens, "is_cld3_available", lambda: False)
    assert "cld3" not in ens.default_backend_order()
    assert len(ens.default_backend_order()) == 6

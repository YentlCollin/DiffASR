# Prétraitement des données

Le point d'entrée public est
`audio.diffusion_data.clean_17k_preprocessing_allowed_shadda`.
Les exports de `audio.diffusion_data` et la préparation conditionnelle utilisent
cette version. L'ancienne implémentation qui retirait aussi les shadda n'est
pas incluse dans ce dossier de publication.

| Option | Résultat |
|---|---|
| `use_diacritics=False` | Conserve les shadda `U+0651` et le tatweel `U+0640` existants ; retire les autres marques arabes configurées. |
| `use_diacritics=True` | Conserve les diacritiques et le tatweel. |
| `use_code_switching=False` | Sélectionne la transcription arabe selon les annotations. |
| `use_code_switching=True` | Utilise les segments code-switch annotés et la consigne correspondante. |

Cette politique n'ajoute aucune shadda ni aucun tatweel. Linguistiquement, la
shadda reste un diacritique ; son traitement ici est une convention des cibles.
Le module gère également les tags de pays, les annotations de transcription,
les cas sans transcription et la validation du format. Voir les signatures
`process_training_transcription` et `process_clean_17k_row` pour les entrées.

La préparation conditionnelle construit les quatre cibles `plain`,
`code_switch`, `diacritics` et `code_switch_diacritics`, leurs longueurs tokenisées
et leurs masques de validité. Les fichiers auxiliaires et l'index portent des
empreintes du code, du tokenizer et du split pour détecter les mélanges.
Le bucket du canvas dépend de la durée audio. Une cible trop longue est
signalée, jamais tronquée ni utilisée pour augmenter silencieusement le bucket.

Un nouveau prétraitement exige une nouvelle destination de métadonnées.
Il ne met pas à jour les anciens Parquet ou les cibles déjà tokenisées.
Les quotas d'epoch et donc le nombre d'updates doivent être recalculés.

Le scorer conserve sa définition historique : `*_without_diacritics` retire
toutes les marques arabes, shadda et tatweel compris, des références et des
hypothèses. Ce choix de métrique est distinct du prétraitement des cibles.
Le score strict conserve ces signes. Comparer les systèmes avec le même
prétraitement, les mêmes références et le même découpage par consigne.

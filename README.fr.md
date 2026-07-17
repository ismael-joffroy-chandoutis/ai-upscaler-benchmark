[English](README.md) · **Français**

# Benchmark des upscalers IA, pour vidéo et image génératives

Un comparatif continuellement mis à jour, sourcé, des upscalers vidéo et image pour
**les images générées par IA**, entre solutions locales (classe RTX 5090) et API, sur
l'arbitrage qualité / coût / vitesse.

**Périmètre.** Vous générez à environ 720p avec des modèles comme LTX-2.3, Wan 2.2,
Veo 3.1, Seedance 2.0 et Kling, et vous voulez monter en 1080p ou 4K. Chaque
recommandation est cadrée pour deux usages distincts : la qualité cinéma d'un côté, le
débit commercial en volume de l'autre. Il n'y a pas de gagnant unique ; la réponse
dépend du plan et du budget.

**Machine de référence** pour les chiffres locaux : NVIDIA RTX 5090 (32 Go GDDR7,
Blackwell), AMD Ryzen 9 9900X, 96 Go DDR5, SSD NVMe 8 To (Samsung 990 Pro) à
~14 000 Mo/s.

> **Statut.** Un comparatif vivant, sourcé en juin 2026 et recoupé. Quatre familles
> sont **mesurées sur la RTX 5090 de référence** : ESRGAN, Topaz classique (Proteus
> 15 fps@1080p / 7,3 fps@4K), SeedVR2 et FlashVSR
> ([data/measurements-5090.json](./data/measurements-5090.json)). Les autres chiffres
> de vitesse sont sourcés et étiquetés avec leur GPU. Voir le [CHANGELOG](./CHANGELOG.md)
> pour les révisions.

---

## L'idée qui organise tout le reste

Votre entrée est **synthétique**. Une image 720p générée par IA n'a pas de vrai grain,
mais elle porte du warping, du flicker temporel, une peau cireuse et un lissage excessif.
Presque tous les upscalers ont été entraînés sur de la vidéo dégradée *réelle*, donc :

- **Les upscalers GAN/CNN accentuent ce qui est déjà là**, en durcissant la peau
  plastique et en amplifiant le flicker.
- **Les upscalers par diffusion régénèrent le détail**, ce qui convient mieux à une
  entrée synthétique, mais ils peuvent inventer un nouveau visage et dériver d'une
  image à l'autre.

Le fait le plus tranchant : l'outil vidéo local ouvert le plus puissant, **SeedVR2**,
porte un avertissement explicite sur sa propre fiche modèle : il sur-accentue la vidéo
générée par IA en 720p. Votre cas d'usage exact est son mode d'échec documenté. Le
correctif n'est pas un autre outil, c'est une main plus légère (force de restauration
plus basse) plus la bonne évaluation, ce qui explique pourquoi ce dépôt pèse sa
méthodologie autant que sa liste d'outils.

---

## Recommandations phares (juin 2026)

**Vidéo** (détail complet dans [docs/video-upscalers.md](./docs/video-upscalers.md)) :

| Usage | Choix local | Choix API |
|---|---|---|
| Cinéma 4K, qualité max | SeedVR2-7B (force basse) | Topaz via fal (Starlight Precise) |
| Volume court-format | FlashVSR Tiny / SeedVR2-3B | WaveSpeed SeedVR2 (~3$/min en 4K) |
| Previews économiques | Real-ESRGAN + `vs_temporalfix` | fal RealESRGAN |
| Sans risque sur les visages, zéro hallucination | Topaz Proteus (Manuel) | Topaz via fal (Proteus) |

**Image** (détail complet dans [docs/image-upscalers.md](./docs/image-upscalers.md)) :

| Usage | Choix local | Choix API |
|---|---|---|
| Plan hero cinéma | SUPIR (v0F) ou Flux-tile | Magnific Precision / Topaz Gigapixel |
| Fidèle, sans invention | ControlNet Tile (SDXL), HAT, DRCT | Topaz Gigapixel (8-12$/100) |
| Traitement par lot économique | Real-ESRGAN (mesuré 1,7 fps fp16, x4 720p, sur la 5090) | fal SeedVR2 (~0,83$/100) |

**Réglages exacts pour chaque modèle** (à copier-coller) : [docs/recipes.md](./docs/recipes.md).
**Prix, valeur, et quoi acheter** (local vs API, l'arbitrage abonnement-vs-usage chez
Topaz, classement des fournisseurs) : [docs/economics.md](./docs/economics.md).

**La règle empirique local-vs-cloud à volume modéré (30 min-3 h/mois) :** l'upscaling
classique/GAN est bien moins cher en local (illimité sur la 5090). La diffusion 4K est
trop lente en local pour du volume (0,2-2 fps), donc la répartition rationnelle est :
carte locale pour la diffusion 1080p plus les plans hero 4K de nuit, cloud pour la 4K
en volume.

---

## Structure du dépôt

```
README.md                  ce fichier (anglais)
README.fr.md                ce fichier (français)
docs/
  video-upscalers.md       le benchmark vidéo
  image-upscalers.md       le benchmark image
  methodology.md           comment benchmarker des upscalers sur entrée synthétique (reproductible)
  recipes.md               réglages exacts, copier-coller, par modèle
  economics.md             prix x qualité, seuil de rentabilité local-vs-API, la décision d'achat Topaz
data/
  registry-schema.json     schéma JSON pour une fiche outil
  tools.json               le registre d'outils lisible par machine (les tableaux en dérivent)
CHANGELOG.md               historique daté de ce qui a changé et a été re-vérifié
CITATION.cff               comment citer
LICENSE.md                 CC BY-NC-ND 4.0 (texte) + PolyForm NC 1.0.0 (code)
```

Les données du benchmark vivent dans `data/tools.json`, pas dans la prose. Les
tableaux Markdown sont une vue lisible de ce registre. C'est délibéré : le tenir à
jour, c'est éditer une petite fiche JSON (et sa date `last_verified`), pas réécrire des
paragraphes.

---

## Comment garder ce dépôt à jour

1. Éditer ou ajouter une fiche dans `data/tools.json` en suivant
   `data/registry-schema.json`. Mettre `last_verified` à aujourd'hui et citer une
   source primaire pour chaque prix ou chiffre dur.
2. Noter le changement dans `CHANGELOG.md` avec la date et la source vérifiée.
3. Tout ce qui ne peut pas être confirmé par une source primaire : le marquer
   `UNVERIFIED` et le garder visible plutôt que de le retirer. La limite de ce qui est
   confirmé est elle-même une information.
4. Quand les mesures de référence 5090 arrivent, remplacer les chiffres de vitesse
   inférés et incrémenter la version mineure.

---

## Méthodologie en un paragraphe

Il n'existe pas de 4K de référence (ground-truth) pour du 720p généré par IA, donc les
métriques standards à référence complète (PSNR/SSIM/LPIPS) ne s'appliquent
généralement pas, et les métriques populaires sans référence (MUSIQ, NIQE, CLIP-IQA+)
**récompensent la sur-accentuation et le détail halluciné que produisent les
upscalers IA**. Le protocole note donc un triptyque par plan (qualité sans référence,
anti-triche fidélité/hallucination, cohérence temporelle) et ne classe jamais sur un
seul chiffre. Protocole complet et reproductible dans
[docs/methodology.md](./docs/methodology.md).

---

## Licence et citation

Texte, tableaux et données : **CC BY-NC-ND 4.0**. Code et schéma :
**PolyForm Noncommercial 1.0.0**. Voir [LICENSE.md](./LICENSE.md). Pour citer, voir
[CITATION.cff](./CITATION.cff).

Maintenu par Ismaël Joffroy Chandoutis. Les contributions de mesures vérifiées
(en particulier de vrais chiffres RTX 5090) et les corrections aux éléments signalés
sont bienvenues via les issues.

Par [Ismaël Joffroy Chandoutis](https://ismaeljoffroychandoutis.com).

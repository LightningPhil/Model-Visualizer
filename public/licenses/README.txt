Third-party source and replacement instructions
==============================================

The application source is MIT licensed:
https://github.com/LightningPhil/Model-Visualizer
This permission includes rebuilding the application with modified libraries.
Reverse engineering for debugging modifications to LGPL-covered components
is permitted. No third-party component is relicensed under the project MIT license.

libavoid-js 0.5.0-beta.5 — LGPL-2.1-or-later
Unmodified upstream JavaScript bindings and separate WebAssembly routing asset.
License: libavoid-LGPL-2.1.txt; full package notices: ../THIRD_PARTY_NOTICES.txt
Pinned binding source and build scripts:
https://github.com/Aksem/libavoid-js/tree/5062a42fbd82fff562afeebcbb7b1ed45eed8e75
Local source archive: sources/libavoid-js-0.5.0-beta.5.tar.gz
Underlying C++ source used by tools/generate.py: Aksem/adaptagrams v1.0.5
https://github.com/Aksem/adaptagrams/tree/0ab3467b4ed3523bb3e404a0dd7da7647d9eb13d
Local source archive: sources/adaptagrams-v1.0.5.tar.gz
To rebuild: extract the bindings, install its npm dependencies, and run
npm run full-rebuild with Python, Git and Docker installed. The upstream
script uses emscripten/emsdk:4.0.7 and clones adaptagrams v1.0.5 into
build/adaptagrams (the supplied C++ archive can be extracted there instead).
To relink: replace node_modules/libavoid-js/dist with the rebuilt JS/WASM,
then run npm run build in Model Visualizer. Vite emits a separate WASM file.
The integration boundary is src/model/avoid.ts and src/model/routing.ts.
Keep these notices with redistributed builds and provide corresponding
source for any modifications you make to the LGPL library.

ELK / elkjs 0.12.0 — used under EPL-2.0
Unmodified upstream graph layout engine; upstream also offers a conditional
secondary GPL option. Original copyright and license text are reproduced
in ../THIRD_PARTY_NOTICES.txt.
Corresponding source and build instructions:
https://github.com/kieler/elkjs/tree/ff5771d7165445c42c408bb8a090c8035272218c
Its Gradle build identifies the Java ELK dependencies and transpilation tools.
https://github.com/eclipse-elk/elk

DM Sans and IBM Plex Mono — SIL Open Font License 1.1
Original copyright and full OFL texts are in ../THIRD_PARTY_NOTICES.txt.
The fonts are bundled locally, not fetched from a font service.

Lucide — ISC, with MIT notices for Feather-derived icons
Both upstream notices are preserved in ../THIRD_PARTY_NOTICES.txt.

Stories260K checkpoint and tokenizer — MIT, Andrej Karpathy
Original unmodified assets:
https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K
Original license: ../models/stories260k/LICENSE
Provenance: ../models/stories260k/README.txt

All other runtime packages and their full license/copyright notices are
listed in ../THIRD_PARTY_NOTICES.txt. The complete runtime and build/test
inventory is docs/dependency-licenses.md in the application repository.
Run npm run licenses after npm ci; production builds regenerate notices.

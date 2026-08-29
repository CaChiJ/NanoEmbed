# M5 Docker Desktop arm64 measurement environment

이 bundle은 M4와 같은 Docker Desktop 4.38.0 계열 VM에서 수집한 동일 환경 비교다.
물리 ARM 장치나 일반 Linux 전체를 대표하지 않는다.

- Docker Engine 27.5.1, Ubuntu 24.04 `linux/arm64`, Linux `6.12.5-linuxkit`
- 8 vCPU, guest RAM 8,217,858,048 bytes, page size 4,096 bytes
- image: `nanoembed-linux-build:latest` (`08db57611abe`)
- source: 현재 repository를 `/src:ro`로 bind mount
- build: `/tmp/nanoembed-m5-linux-build`를 `/build`로 bind mount
- tokenizer cache: `/tmp/nanoembed-m5-cache`를 `/cache`로 bind mount
- compiler: GCC/G++ 13.3.0, Release `-O3 -DNDEBUG`, OpenMP, `GGML_NATIVE=ON`
- NanoEmbed HEAD: `271d8465b042a4b19eaf85cc5066d2eed0a19374` + 의도된 dirty M5 worktree
- ggml: `387fa29fbbf3149f06a631c7850b6c35c24b0232` (서브모듈 수정 없음)
- benchmark binary SHA-256: `1b2b1366f6a8d40eb5c604ec1978e5705aaa0c18f1da52fa5dd597611b6c068d`
- scenario SHA-256: `51d150ab5b4b642184deb14fca7118ce17bad345c83f6f310f779c764d512465`
- corpus manifest SHA-256: `bcbe20d6e58dc42fbf4944cc9d196b3d654c6ef3740d29cb49eb6db7a2fac863`
- selection seed: 0
- mixed-short N=32 selection: `af374fc457de85a1a5e63fa88907c54672d9854ac79c8a169c67109f3a91f7bc`
- mixed-short N=128 selection: `0897734582c81b752dacdab385765b76701e0fb081ba57f455834e7625739f24`

profile-off latency/throughput만 성능 판정에 사용했다. profile-on은 10ms
`smaps_rollup` 관측이며 latency는 diagnostic, PSS/USS peak는 sampled lower bound다.


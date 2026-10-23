"""Pure shadow fitting math; real GPU coverage is checked in the browser suite."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
MODULE = (Path(__file__).resolve().parents[1] / "src/robotforge/viewer/static/lighting-fit.js").as_uri()
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_js(body):
    script = f"import assert from 'node:assert/strict'; import {{fitShadowBounds}} from {json.dumps(MODULE)};\n{body}"
    result = subprocess.run(
        [NODE, "--experimental-default-type=module", "--input-type=module", "-"],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "bounds",
    [
        {"min": [0, 0, 0], "max": [0, 0, 0]},
        {"min": [-0.001, 0, -0.002], "max": [0.001, 0.001, 0.002]},
        {"min": [-1, -2, -3], "max": [2, 3, 4]},
        {"min": [-40, -8, -2], "max": [30, 16, 40]},
        {"min": [-1000, -900, -600], "max": [2000, 1400, 900]},
        {"min": [1e7, -1e7, 300], "max": [1e7 + 0.01, -1e7 + 0.02, 300.01]},
    ],
)
def test_fit_contains_tiny_large_and_translated_bounds(bounds):
    run_js(f"""const b={json.dumps(bounds)}, f=fitShadowBounds(b);
      assert(f); assert(f.near>0&&f.far>f.near);
      const lo=[f.x-f.width/2,f.y-f.height/2,f.z-f.far],
        hi=[f.x+f.width/2,f.y+f.height/2,f.z-f.near];
      for(let i=0;i<3;i++){{assert(lo[i]<=b.min[i]);assert(hi[i]>=b.max[i]);}}
      assert(f.normalBias>=.00015&&f.normalBias<=.004);
      assert(f.bias<0&&Math.abs(f.bias)*(f.far-f.near)<=.00100001);
      assert(Math.abs(f.x/(f.width/2048)-Math.round(f.x/(f.width/2048)))<1e-5);
      assert(Math.abs(f.y/(f.height/2048)-Math.round(f.y/(f.height/2048)))<1e-5);
    """)


def test_small_motion_reuses_fit_but_departure_and_reset_refit():
    run_js("""const b={min:[0,0,0],max:[1,1,1]}, first=fitShadowBounds(b);
      assert.equal(fitShadowBounds({min:[.001,.001,0],max:[1.001,1.001,1]},first),first);
      const away=fitShadowBounds({min:[4,0,0],max:[5,1,1]},first); assert.notEqual(away,first);
      const reset=fitShadowBounds(b); assert.deepEqual(reset,first);
      assert.equal(fitShadowBounds(b,reset),reset);
    """)


def test_bad_or_empty_bounds_preserve_no_invalid_projection():
    run_js("""for(const b of [{min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]},
      {min:[NaN,0,0],max:[1,1,1]}, {min:[1,0,0],max:[0,1,1]}]) assert.equal(fitShadowBounds(b),null);
      for(const n of [0,1,NaN,16.5]) assert.throws(()=>fitShadowBounds({min:[0,0,0],max:[1,1,1]},null,n));
    """)


def test_seeded_bounds_and_movement_always_remain_inside_the_actual_frustum():
    run_js("""let seed=17, previous=null; const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/2**32};
      for(let k=0;k<1000;k++){
        const min=Array.from({length:3},()=>rand()*200-100), max=min.map(v=>v+10**(rand()*5-3));
        const f=fitShadowBounds({min,max},previous), lo=[f.x-f.width/2,f.y-f.height/2,f.z-f.far],
          hi=[f.x+f.width/2,f.y+f.height/2,f.z-f.near];
        for(let i=0;i<3;i++){assert(lo[i]<=min[i]);assert(hi[i]>=max[i]);}
        previous=f;
      }
    """)

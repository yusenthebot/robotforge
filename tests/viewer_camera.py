"""Bounded observation of actual camera motion before projecting native input."""

from __future__ import annotations


async def settle_camera(tab, *, quiet_ms: int = 0) -> None:
    await tab.run(
        """
      let previous=null, stable=0, since=performance.now(), diagnostic=null;
      const deadline=performance.now()+6000;
      while(performance.now()<deadline){
        await new Promise(requestAnimationFrame);
        const rf=window.__robotforge,c=rf.three().camera;
        const state=[...c.projectionMatrix.elements,...c.position.toArray(),...rf.cameraTarget()];
        const delta=previous?Math.max(...state.map((v,i)=>Math.abs(v-previous[i]))):Infinity;
        const ready=rf.renderedVersion===rf.scene?.version&&!rf.tweening();
        // OrbitControls reconstructs coordinates each frame, leaving observed 1e-15 roundoff at rest.
        if(ready&&delta<=1e-9)stable++;
        else{stable=0;since=performance.now()}
        previous=state;
        diagnostic={delta,ready,stable,time:rf.time};
        if(stable>=8&&performance.now()-since>=QUIET_MS)return;
      }
      throw Error('camera and projection did not settle: '+JSON.stringify(diagnostic));
    """.replace("QUIET_MS", str(quiet_ms))
    )

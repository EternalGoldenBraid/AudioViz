export default async function checkTransformProgression(page) {
  const check = (condition,message) => { if (!condition) throw new Error(message); };
  const panel = page.getByRole("region",{name:"2D transform progression"});
  const spatial = page.getByRole("region",{name:"3D transform progression"});
  const positions = selector => panel.locator(selector).evaluateAll(nodes =>
    nodes.map(p => [p.getAttribute("cx"),p.getAttribute("cy")]));
  await panel.waitFor();
  check(await panel.locator("[data-scene-mode]").count() === 0,"The 2D section contains only image correspondences");
  check(await spatial.locator(".image-viewport").count() === 0,"The 3D section contains only the world construction");
  const source = await positions("[data-transform-source]");
  const original = await page.locator('[data-scene-mode="world"] svg').innerHTML();
  const world = spatial.locator('[data-scene-mode="progression-world"]');
  const initialWorld = await world.locator("svg").innerHTML();
  for (let stage=1;stage<=4;stage++) {
    await panel.locator(`[data-transform-stage="${stage}"]`).click();
    check(await panel.getAttribute("data-transform-family") === String(stage-1),`Milestone ${stage} selects its exact family`);
    const outputs = await panel.locator(".model-errors output").allTextContents();
    outputs.forEach((value,i) => check((value === "exact") === (i >= stage-1),`Milestone ${stage}: model ${i} exactness`));
    check(JSON.stringify(await positions("[data-transform-source]")) === JSON.stringify(source),"Source rays stay fixed");
    check(await world.locator("svg").innerHTML() === initialWorld,"The 2D slider leaves the independent 3D view unchanged");
  }
  check(await page.locator('[data-scene-mode="world"] svg').innerHTML() === original,"The experiment leaves the main world controls intact");
  check(JSON.stringify(await positions("[data-transform-target]")) !== JSON.stringify(await positions("[data-transform-predicted]")),
    "Off-plane samples separate from plane predictions");
  const plot = await panel.locator(".image-viewport svg").innerHTML();
  for (let stage=1;stage<=4;stage++) {
    await spatial.locator(`[data-transform-stage="${stage}"]`).click();
    check(await spatial.getAttribute("data-transform-family") === String(stage-1),`3D milestone ${stage} selects its exact family`);
    check(await panel.locator(".image-viewport svg").innerHTML() === plot,"The 3D slider leaves the independent 2D view unchanged");
  }
  check(await world.locator("svg").innerHTML() !== initialWorld,"The 3D slider changes the world geometry");
  await panel.locator("#transform-complexity-2d").focus();
  await page.keyboard.press("Home");
  check(await panel.getAttribute("data-transform-family") === "0","The slider supports keyboard reset");
  await spatial.locator("#transform-complexity-3d").focus();
  await page.keyboard.press("Home");
  check(await spatial.getAttribute("data-transform-family") === "0","The 3D slider supports keyboard reset");
  check(await world.locator('[data-point^="construction-"]').count() === 25,"The 3D construction contains every image sample");
  await world.getByRole("button",{name:"Reset view"}).click();
  const resetPlot = await panel.locator(".image-viewport svg").innerHTML();
  await page.locator("#reader-azimuth").evaluate(input => {
    input.value = "-35";
    input.dispatchEvent(new Event("input",{bubbles:true}));
  });
  check(await panel.locator(".image-viewport svg").innerHTML() === resetPlot,"Reader motion does not alter image correspondences");
  await world.getByRole("button",{name:"Reset view"}).click();
}
